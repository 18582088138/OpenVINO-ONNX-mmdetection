import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import openvino as ov
from openvino import Core
import time

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

class SeparateGroundingDINOOpenVINO:
    """OpenVINO Separate GroundingDINO Pipeline"""
    def __init__(self, language_model_path, visual_model_path, transformer_path, 
                 bert_model_path=None, device='CPU'):
        self.core = Core()
        self.language_model = self.core.compile_model(str(language_model_path), device)
        self.visual_model = self.core.compile_model(str(visual_model_path), device)
        self.transformer_model = self.core.compile_model(str(transformer_path), device)
        
        # 使用默认的 BERT 路径，与转换脚本保持一致
        if bert_model_path is None:
            script_dir = Path(__file__).resolve().parent
            bert_model_path = script_dir / 'models' / 'bert-base-uncased'
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.cfg = {
                'max_tokens': 256,
                'special_tokens_list': ['[CLS]', '[SEP]', '.', '?'],
                'max_per_img':  300
            }
        print("[INFO] OpenVINO Model Init Done")

    def preprocess_text(self, text_prompts, max_tokens=256, pad_to_max=False):
        tokenized = self.tokenizer.batch_encode_plus(
            text_prompts,
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

def main():
    script_dir = Path(__file__).resolve().parent
    base_name = f"gdino_swinb_800_1333"
    ir_model_dir = script_dir / 'models' / 'IR_model'
    
    language_model_path = ir_model_dir / f"{base_name}_language.xml"
    visual_model_path = ir_model_dir / f"{base_name}_visual.xml"
    transformer_path = ir_model_dir / f"{base_name}_transformer.xml"
    
    # 检查模型文件是否存在
    for model_path in [language_model_path, visual_model_path, transformer_path]:
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_path}")
            print("[INFO] Please run ov_convert_groundingdino_pipeline.py first to generate the IR models")
            return
    
    inferencer = SeparateGroundingDINOOpenVINO(
        language_model_path, visual_model_path, transformer_path,
        device='CPU')
    
    batch_size = 1
    height, width = 800, 1333
    img = np.random.randn(batch_size, 3, height, width).astype(np.float32)
    classes = ('Face', 'Person', 'Pet', 'Vehicle', 'Plate', 'Nonmotor', 'Head')
    text_prompts = [' . '.join(classes) + ' .']
    print("[INFO] OV GroundingDINO Pipeline Infer Start...")
    cls_scores, bbox_preds = inferencer.inference(img, text_prompts)
    print(f"[INFO] OV GroundingDINO Pipeline Infer Done")
    # print(f"[INFO] OV GroundingDINO cls_scores: {cls_scores[0][:5]}")
    print(f"[INFO] OV GroundingDINO bbox_preds: \n{bbox_preds[0][:5]}")

if __name__ == '__main__':
    main() 