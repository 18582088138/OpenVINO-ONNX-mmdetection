import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import openvino as ov
from openvino import Core
import time
import cv2
import re
import torch
import supervision as sv
from argparse import ArgumentParser

from PIL import Image
from torchvision import transforms
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

def preprocess_image(image_path, shape=(512,512)):
    img = Image.open(image_path)
    transformers = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transformers(img)


def preprocess_images(image_paths, shape=(512,512)):
    images = []
    for path in image_paths:
        images.append(preprocess_image(path, shape))

    return torch.stack(images, dim=0)

def generate_masks_with_special_tokens_and_transfer_map_np(
        tokenized, special_tokens_list):
    """
    [Using Numpy instead of PyTorch]
    [Reference] from mmdet.models.language_models.bert import generate_masks_with_special_tokens_and_transfer_map
    Generate attention mask between each pair of special tokens.
    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (np.ndarray): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(np.ndarray, np.ndarray):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenever a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.argwhere(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token, dtype=bool)[None, :, :].repeat(bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=int)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = np.arange(
                0, col - previous_col)
        previous_col = col

    return attention_mask, position_ids

def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    nltk_dir = SCRIPT_DIR / 'models' / 'nltk_data'
    try:
        import nltk
        nltk_dir.mkdir(parents=True, exist_ok=True)
        nltk.download('punkt', download_dir=nltk_dir)
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_dir)
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: '
                           'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punctuation:
        text = text.replace(p, '')
    return text.strip()

def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print('noun entities:', noun_phrases)
            print('entity:', entity)
            print('caption:', caption.lower())
    return tokens_positive, noun_phrases

def create_positive_map(tokenized,
                        tokens_positive: list,
                        max_num_entities: int = 256):
    """construct a map such that positive_map[i,j] = True
    if box i is associated to token j

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                               dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print('beg:', beg, 'end:', end)
                print('token_positive:', tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token(positive_map,
                                       plus: int = 0) -> dict:
    """Create a dictionary mapping the label to the token.
    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def get_positive_map(text_prompts, tokenizer, cfg):
    tokens_positive, entities = run_ner(text_prompts)
    tokenized = tokenizer.batch_encode_plus(
            [text_prompts],
            max_length=cfg['max_tokens'],
            padding='longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True)
    positive_map = create_positive_map(tokenized, tokens_positive)
    positive_map_label_to_token = create_positive_map_label_to_token(positive_map, plus=1)
    return positive_map_label_to_token, entities

class OVSeparateGroundingDINO:
    """OpenVINO Separate GroundingDINO Pipeline"""
    def __init__(self, language_model_path, visual_model_path, transformer_path, 
                 bert_model_path=None, device='CPU'):
        self.core = Core()
        self.language_model = self.core.compile_model(str(language_model_path), device)
        self.visual_model = self.core.compile_model(str(visual_model_path), device)
        self.transformer_model = self.core.compile_model(str(transformer_path), device)
        
        if bert_model_path is None:
            bert_model_path = SCRIPT_DIR / 'models' / 'bert-base-uncased'
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.cfg = {
                'max_tokens': 256,
                'special_tokens_list': ['[CLS]', '[SEP]', '.', '?'],
                'max_per_img':  300
            }
        print("[INFO] OpenVINO Model Init Done")

    def preprocess_text(self, text_prompts, max_tokens=256, pad_to_max=False):
        tokenized = self.tokenizer.batch_encode_plus(
            [text_prompts],
            max_length=max_tokens,
            padding='max_length' if pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='np',
            truncation=True)
        
        special_tokens = self.tokenizer.convert_tokens_to_ids(self.cfg['special_tokens_list'])
        attention_mask, position_ids = generate_masks_with_special_tokens_and_transfer_map_np(tokenized, special_tokens)
        
        input_ids = tokenized['input_ids']
        attention_mask = attention_mask
        position_ids = position_ids
        token_type_ids = np.zeros_like(input_ids)
        return input_ids, attention_mask, position_ids, token_type_ids

    def run_language_model(self, input_ids, attention_mask, position_ids, token_type_ids):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
        }
        start = time.time()
        result = self.language_model(inputs)
        elapsed = time.time() - start
        self.language_time = elapsed
        keys = list(result.keys())
        embedded = result[keys[0]]
        masks = result[keys[1]]
        hidden = result[keys[2]]
        return embedded, masks, hidden

    def run_visual_model(self, img):
        inputs = {'img': img}
        start = time.time()
        result = self.visual_model(inputs)
        elapsed = time.time() - start
        self.visual_time = elapsed
        keys = list(result.keys())
        visual_feat_0 = result[keys[0]]
        visual_feat_1 = result[keys[1]]
        visual_feat_2 = result[keys[2]]
        visual_feat_3 = result[keys[3]]
        return visual_feat_0, visual_feat_1, visual_feat_2, visual_feat_3

    def run_transformer(self, visual_feat_0, visual_feat_1, visual_feat_2, visual_feat_3, 
                                embedded, masks, position_ids, text_token_mask):
        inputs = {
            "visual_feat_0":visual_feat_0,
            "visual_feat_1":visual_feat_1,
            "visual_feat_2":visual_feat_2,
            "visual_feat_3":visual_feat_3, 
            'embedded': embedded,
            'masks': masks,
            'position_ids': position_ids,
            'text_token_mask': text_token_mask,
        }
        start = time.time()
        result = self.transformer_model(inputs)
        elapsed = time.time() - start
        self.transformer_time = elapsed
        keys = list(result.keys())
        keys = list(result.keys())
        return result[keys[0]], result[keys[1]]


    def inference(self, img, text_prompts, max_tokens=256):
        input_ids, attention_mask, position_ids, token_type_ids = self.preprocess_text(
            text_prompts, max_tokens)
        embedded, masks, hidden = self.run_language_model(input_ids, attention_mask, position_ids, token_type_ids)
        visual_feats = self.run_visual_model(img)
        batch_size, seq_len = input_ids.shape
        text_token_mask = np.ones((batch_size, seq_len), dtype=bool)
        cls_scores, bbox_preds = self.run_transformer(
            visual_feats[0], visual_feats[1], visual_feats[2], visual_feats[3], 
            embedded, masks, position_ids, text_token_mask)
    
        print(f"[INFO] Language Embedding Infer time cost: {self.language_time*1000:.2f} ms")
        print(f"[INFO] Vision Embedding Infer time cost: {self.visual_time*1000:.2f} ms")
        print(f"[INFO] Transformer Infer time cos: {self.transformer_time*1000:.2f} ms")
        return cls_scores, bbox_preds

# 后处理函数，参考 ov_infer_e2e.py

def convert_grounding_to_cls_scores(logits, positive_maps):
    # logits: (B, N, L), positive_maps: list of dict
    assert len(positive_maps) == logits.shape[0]  # batch size
    scores = np.zeros((logits.shape[0], logits.shape[1], len(positive_maps[0])))
    if all(x == positive_maps[0] for x in positive_maps):
        positive_map = positive_maps[0]
        for label_j in positive_map:
            scores[:, :, label_j - 1] = logits[:, :, positive_map[label_j]].mean(-1)
    else:
        for i, positive_map in enumerate(positive_maps):
            for label_j in positive_map:
                scores[i, :, label_j - 1] = logits[i, :, positive_map[label_j]].mean(-1)
    return scores

def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = np.split(bbox, 4, axis=-1)
    bbox_new = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return np.concatenate(bbox_new, axis=-1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def annotate_once(image_path, output_path, classes, cls_score, bbox_pred, token_positive_map, threshold=0.3):
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape
    # 先mean再sigmoid，保证与E2E一致
    cls_score = convert_grounding_to_cls_scores(
        logits=cls_score[None],
        positive_maps=[token_positive_map])[0]
    cls_score = sigmoid(cls_score)
    scores = cls_score.reshape(-1)
    indexes = np.argsort(-scores)[:300]
    num_classes = cls_score.shape[-1]
    det_labels = indexes % num_classes
    bbox_index = indexes // num_classes
    bbox_pred = bbox_pred[bbox_index]
    det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
    det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * w
    det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * h
    idx = scores[indexes] > threshold
    detections = sv.Detections(xyxy=det_bboxes[idx], confidence=scores[indexes][idx], class_id=det_labels[idx])
    labels = []
    for j in range(len(detections)):
        class_name = classes[detections.class_id[j]]
        confidence = detections.confidence[j]
        labels.append(f"{class_name} {confidence*100:.1f}")
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator(text_scale=0.4)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    cv2.imwrite(str(output_path), annotated_image)

# 命令行参数解析

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../demo/large_image.jpg', help="Input images (comma separated)")
    parser.add_argument('-o', '--outdir', type=str, default='./outputs', help="Output directory")
    parser.add_argument('-p', '--prompt', type=str, default='car', help="Prompt")
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help="Score threshold")
    parser.add_argument('-lm', '--language_model', type=str, default=None, help="OpenVINO language model xml")
    parser.add_argument('-vm', '--visual_model', type=str, default=None, help="OpenVINO visual model xml")
    parser.add_argument('-tm', '--transformer_model', type=str, default=None, help="OpenVINO transformer model xml")
    parser.add_argument('-bm', '--bert_model', type=str, default=None, help="BERT model path")
    return parser.parse_args()

def main():
    args = parse_args()
    base_name = f"gdino_swinb_800_1333"
    ir_model_dir = SCRIPT_DIR / 'models' / 'IR_model'
    # OV model init
    language_model_path = args.language_model or (ir_model_dir / f"{base_name}_language.xml")
    visual_model_path = args.visual_model or (ir_model_dir / f"{base_name}_visual.xml")
    transformer_path = args.transformer_model or (ir_model_dir / f"{base_name}_transformer.xml")
    bert_model_path = args.bert_model or (SCRIPT_DIR / 'models' / 'bert-base-uncased')
    for model_path in [language_model_path, visual_model_path, transformer_path]:
        if not Path(model_path).exists():
            print(f"[ERROR] Model file not found: {model_path}")
            print("[INFO] Please run ov_convert_groundingdino_pipeline.py first to generate the IR models")
            return
    ov_inferencer = OVSeparateGroundingDINO(
                        language_model_path, visual_model_path, transformer_path,
                        bert_model_path=bert_model_path, device='CPU')
    # 处理图片路径
    images_path = Path(args.images)
    if images_path.is_dir():
        # 支持常见图片格式
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(sorted(str(p) for p in images_path.glob(ext)))
    elif ',' in args.images:
        image_paths = [x.strip() for x in args.images.split(',')]
    else:
        image_paths = [args.images]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    batch_size = len(image_paths)

    # image preprocess
    img_shape = (ov_inferencer.visual_model.inputs[0].partial_shape[2].get_max_length(),
                 ov_inferencer.visual_model.inputs[0].partial_shape[3].get_max_length())
    print(f"[INFO] img_shape: {img_shape}")
    batch_imgs = preprocess_images(image_paths, img_shape)

    # prompt preprocess
    if args.prompt is None:
        classes = ('Face', 'Person', 'Pet', 'Vehicle', 'Plate', 'Nonmotor', 'Head')
        text_prompts = ' . '.join(classes) + ' .'
    else:
        text_prompts = args.prompt    
    token_positive_map, classes = get_positive_map(text_prompts, ov_inferencer.tokenizer, ov_inferencer.cfg)
    print(f"[INFO] classes: {classes}")

    # OV inference
    print("[INFO] OV GroundingDINO Pipeline Infer Start...")
    cls_scores, bbox_preds = ov_inferencer.inference(batch_imgs, text_prompts)
    print(f"[DEBUG] OV GroundingDINO cls_scores: \n{cls_scores[0][:5]}")
    print(f"[DEBUG] OV GroundingDINO bbox_preds: \n{bbox_preds[0][:5]}")

    # postprocess and save result
    for i, img_path in enumerate(image_paths):
        output_file = outdir / Path(img_path).name
        annotate_once(img_path, output_file, classes, cls_scores[i], bbox_preds[i], token_positive_map, args.threshold)
        print(f"[INFO] Saved result to {output_file}")
    print(f"[INFO] OV GroundingDINO Pipeline Infer Done")

if __name__ == '__main__':
    main() 