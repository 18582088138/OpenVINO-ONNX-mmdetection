import cv2
import re
import time
import torch
import openvino as ov
import supervision as sv

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import List, Tuple
from transformers import AutoTokenizer
from torchvision import transforms


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


def generate_masks_with_special_tokens(tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token))
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

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
    print('noun_phrases:', noun_phrases)
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


def get_positive_map(text_prompts: str):
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

def infer_once(model, batch_img, input_ids, position_ids, attention_mask):
    inputs = {}
    #input_img = np.expand_dims(image, 0)
    inputs["img"] = batch_img
    inputs["input_ids"] = input_ids
    inputs["text_token_mask"] = attention_mask
    inputs["position_ids"] = position_ids
    outs = model.infer_new_request(inputs)

    return torch.Tensor(outs['cls']), torch.Tensor(outs['coords'])

def convert_grounding_to_cls_scores(logits: torch.Tensor,
                                    positive_maps: List[dict]) -> torch.Tensor:
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(logits.shape[0], logits.shape[1],
                         len(positive_maps[0])).to(logits.device)

    if all(x == positive_maps[0] for x in positive_maps):
        # only need to compute once
        positive_map = positive_maps[0]
        for label_j in positive_map:
            scores[:, :, label_j -
                   1] = logits[:, :,
                               torch.LongTensor(positive_map[label_j]
                                                )].mean(-1)
    else:
        for i, positive_map in enumerate(positive_maps):
            for label_j in positive_map:
                scores[i, :, label_j - 1] = logits[
                    i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    return scores


def bbox_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def annotate_once(image_path, output_path, classes, cls_score, bbox_pred, token_positive_map, threshold=0.3):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    cls_score = convert_grounding_to_cls_scores(
        logits=cls_score.sigmoid()[None],
        positive_maps=[token_positive_map])[0]
    scores, indexes = cls_score.view(-1).topk(cfg['max_per_img'])
    num_classes = cls_score.shape[-1]
    det_labels = indexes % num_classes
    bbox_index = indexes // num_classes
    bbox_pred = bbox_pred[bbox_index]

    det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
    det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * w
    det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * h

    idx = scores > threshold
    detections = sv.Detections(xyxy=det_bboxes[idx].numpy(), confidence=scores[idx].numpy(), class_id=det_labels[idx].numpy())
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
    # save the annotated grounded-sam image
    cv2.imwrite(output_path, annotated_image)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        default='./models/IR_model/gdino_swinb_800_1333.xml',
                        help="OpenVINO *.xml file")
    parser.add_argument('-d', '--device', type=str, default='CPU', help="Device to use")
    parser.add_argument('-p', '--prompt', type=str, default='car', help="prompt")
    parser.add_argument('-i', '--images', type=str, default='../demo/large_image.jpg', help="Input images")
    parser.add_argument('-o', '--outdir', type=str, default='./outputs',help="Output directory")
    parser.add_argument('-t', '--threshold', type=float, default=0.2)

    args = parser.parse_args()
    return args

cfg = {
    'name': str(SCRIPT_DIR / 'models' / 'bert-base-uncased'),
    'max_tokens': 256,
    'special_tokens_list': ['[CLS]', '[SEP]', '.', '?'],
    'max_per_img':  300
}
tokenizer = AutoTokenizer.from_pretrained(cfg['name'])

def main():
    args = parse_args()

    if args.images.find(',') > 0:
        image_paths = args.images.split(',')
    else:
        image_paths = [args.images]
    bs = len(image_paths)

    model = ov.compile_model(args.model, args.device)

    img_shape = (model.inputs[0].partial_shape[2].get_max_length(),
                 model.inputs[0].partial_shape[3].get_max_length())
    print(img_shape)
    #img_shape = (args.height, args.width)
    batch_img = preprocess_images(image_paths, img_shape)

    text_prompts = args.prompt

    token_positive_map, classes = get_positive_map(text_prompts)

    tokenized = tokenizer.batch_encode_plus(
            [text_prompts] * bs,
            max_length=cfg['max_tokens'],
            padding='longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True)

    special_tokens = tokenizer.convert_tokens_to_ids(cfg['special_tokens_list'])
    attention_mask, position_ids = generate_masks_with_special_tokens(tokenized, special_tokens)

    cls, bbox = infer_once(model, batch_img, tokenized['input_ids'], position_ids, attention_mask)
    print("[Warm up]")
    start_time = time.time()
    infer_count = 1
    cls, bbox = infer_once(model, batch_img, tokenized['input_ids'], position_ids, attention_mask)
    print(f"[OpenVINO] Inference time cost: {((time.time()-start_time)/infer_count)*1000:.2f} ms")

    assert(cls.shape[0] == bs and bbox.shape[0] == bs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(bs):
        output_file = outdir / image_paths[i].split('/')[-1]
        print(f"to annotate {output_file}")
        annotate_once(image_paths[i], output_file, classes, cls[i], bbox[i], token_positive_map, args.threshold)


if __name__ == '__main__':
    main()
