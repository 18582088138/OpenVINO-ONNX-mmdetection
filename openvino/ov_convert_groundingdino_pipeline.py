import torch
from argparse import ArgumentParser
from pathlib import Path
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from transformers import AutoTokenizer
from mmdet.models.language_models.bert import generate_masks_with_special_tokens_and_transfer_map

import openvino as ov
from openvino import Core

core = Core()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default="../configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py", help='Config file')
    parser.add_argument('--weights', type=str, default="./models/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth", help='*.pth file')
    parser.add_argument('--h', type=str, default="800", help='input width')
    parser.add_argument('--w', type=str, default="1333", help='input height')

    args = parser.parse_args()
    return args

class LanguageModelWrapper(torch.nn.Module):
    """Language wrapper for model export """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask, position_ids, token_type_ids):
        text_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
        }
        text_feat = self.model.language_model.language_backbone(text_inputs)
        print("[DEBUG] text_feat keys:", text_feat.keys())
        print("[DEBUG] embedded shape:", text_feat['embedded'].shape)
        print("[DEBUG] masks shape:", text_feat['masks'].shape)
        print("[DEBUG] hidden shape:", text_feat['hidden'].shape)
        text_feat['embedded'] = self.model.text_feat_map(text_feat['embedded'])
        return text_feat['embedded'], text_feat['masks'], text_feat['hidden']


class VisualModelWrapper(torch.nn.Module):
    """Vision wrapper for model export """
    def __init__(self, model, img_shape):
        super().__init__()
        self.model = model
        self.img_meta = dict(
            img_shape=img_shape,
            pad_shape=img_shape,
            batch_input_shape=img_shape,
        )
        
    def forward(self, img):
        visual_feats = self.model.extract_feat(img)
        return visual_feats


class TransformerWrapper(torch.nn.Module):
    """ Transformer wrapper for model export """
    def __init__(self, model, img_shape):
        super().__init__()
        self.model = model
        self.img_meta = dict(
            img_shape=img_shape,
            pad_shape=img_shape,
            batch_input_shape=img_shape,
        )
        
    def forward(self, visual_feat_0, visual_feat_1, visual_feat_2, visual_feat_3, 
                        embedded, masks, position_ids, text_token_mask):
        text_feats = {
            'embedded': embedded,
            'masks': masks,
            'position_ids': position_ids,
            'text_token_mask': text_token_mask,
        }
        bs = visual_feat_0[0].shape[0] 
        visual_feats = [visual_feat_0, visual_feat_1, visual_feat_2, visual_feat_3]
        batched_data_samples = []
        for i in range(bs):
            data_sample = DetDataSample(metainfo=self.img_meta)
            batched_data_samples.append(data_sample)
        head_inputs_dict = self.model.forward_transformer(visual_feats, text_feats, batched_data_samples)
        
        outs = self.model.bbox_head(
            memory_text=head_inputs_dict["memory_text"],
            text_token_mask=head_inputs_dict["text_token_mask"],
            hidden_states=head_inputs_dict["hidden_states"],
            references=head_inputs_dict["references"])
        return outs[0][-1], outs[1][-1]

def export_language_model(model, tokenizer, cfg, output_path):
    classes = ('Face', 'Person', 'Pet', 'Vehicle', 'Plate', 'Nonmotor', 'Head')
    text_prompts = [' . '.join(classes) + ' .']
    
    tokenized = tokenizer.batch_encode_plus(
        text_prompts,
        max_length=cfg.max_tokens,
        padding='max_length' if cfg.pad_to_max else 'longest',
        return_special_tokens_mask=True,
        return_tensors='pt',
        truncation=True)
    
    text_token_mask, position_ids = \
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, model.language_model.special_tokens)
    print("[DEBUG] special_tokens:", model.language_model.special_tokens)
    input_ids = tokenized['input_ids']
    attention_mask = text_token_mask
    position_ids = position_ids
    token_type_ids = torch.zeros_like(input_ids)
        
    lang_model_wrapper = LanguageModelWrapper(model).eval()
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len_2"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "embedded": {0: "batch_size", 1: "seq_len"},
        "masks": {0: "batch_size", 1: "seq_len", 2: "seq_len_2"},
        "hidden": {0: "batch_size", 1: "seq_len"},
    }
    
    with torch.no_grad():
        torch.onnx.export(
            lang_model_wrapper,
            args=(input_ids, attention_mask, position_ids, token_type_ids),
            f=output_path,
            input_names=["input_ids", "attention_mask", "position_ids", "token_type_ids"],
            output_names=["embedded", "masks", "hidden"],
            dynamic_axes=dynamic_axes,
            opset_version=16)
    print(f"[INFO] ONNX language model convert : {output_path}")

    ov_input = {"input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "position_ids": position_ids, 
                "token_type_ids": token_type_ids}
    ov_save_path = output_path.parent.parent / 'IR_model' / output_path.name.replace('.onnx', '.xml')
    ov_save_path.parent.mkdir(parents=True, exist_ok=True)
    # ov_model = ov.convert_model(str(output_path), example_input=ov_input)
    ov_model = ov.convert_model(lang_model_wrapper, example_input=ov_input)
    ov.save_model(ov_model, str(ov_save_path))
    print(f"[INFO] OpenVINO language model convert : {ov_save_path}")


def export_visual_model(model, img_shape, output_path):   
    batch_inputs = torch.randn(1, 3, *img_shape)

    visual_model_wrapper = VisualModelWrapper(model, img_shape).eval()
    dynamic_axes = {
        "img": {0: "batch_size"},
        "visual_feat_0": {0: "batch_size"},
        "visual_feat_1": {0: "batch_size"},
        "visual_feat_2": {0: "batch_size"},
        "visual_feat_3": {0: "batch_size"},
    }
    
    with torch.no_grad():
        torch.onnx.export(
            visual_model_wrapper,
            args=(batch_inputs,),
            f=output_path,
            input_names=["img"],
            output_names=["visual_feat_0", "visual_feat_1", "visual_feat_2", "visual_feat_3"],
            dynamic_axes=dynamic_axes,
            opset_version=16)
    
    print(f"[INFO] ONNX vision model convert : {output_path}")

    ov_input = {"img": batch_inputs}
    ov_input_name =  {"img": [1,3,800,1333]}
    ov_save_path = output_path.parent.parent / 'IR_model' / output_path.name.replace('.onnx', '.xml')
    ov_save_path.parent.mkdir(parents=True, exist_ok=True)
    # ov_model = ov.convert_model(str(output_path), example_input=ov_input)
    ov_model = ov.convert_model(visual_model_wrapper, input=ov_input_name, example_input=ov_input)
    ov.save_model(ov_model, str(ov_save_path))
    print(f"[INFO] OpenVINO vision model convert : {ov_save_path}")


def export_transformer(model, img_shape, output_path):    
    img = torch.randn(1, 3, *img_shape)
    # 使用短文本，确保attention_mask有False（有padding）
    classes = ('Face', 'Person', 'Pet', 'Vehicle', 'Plate', 'Nonmotor', 'Head')
    text_prompts = [' . '.join(classes) + ' .']
    
    cfg = model.language_model_cfg
    # 使用本地路径，避免网络连接问题
    bert_path = Path(__file__).resolve().parent / 'models' / 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    tokenized = tokenizer.batch_encode_plus(
                text_prompts,
                max_length=cfg.max_tokens,
                padding='max_length' if cfg.pad_to_max else 'longest',
                return_special_tokens_mask=True,
                return_tensors='pt',
                truncation=True)
    print("[DEBUG] attention_mask for export:", tokenized['attention_mask'])
    
    text_token_mask, position_ids = \
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, model.language_model.special_tokens)

    input_ids = tokenized['input_ids']
    text_inputs = {
        'input_ids': input_ids,
        'attention_mask': text_token_mask,
        'position_ids': position_ids,
        'token_type_ids': torch.zeros_like(input_ids),
    }
    text_feat = model.language_model.language_backbone(text_inputs)
    print("[DEBUG] text_feat keys:", text_feat.keys())
    
    text_feat['embedded'] = model.text_feat_map(text_feat['embedded'])
    text_feat['position_ids'] = position_ids
    text_feat['text_token_mask'] = torch.ones_like(input_ids).bool()
    print(f"embedded shape: {text_feat['embedded'].shape}, type: {text_feat['embedded'].dtype}")
    print(f"masks shape: {text_feat['masks'].shape}, type: {text_feat['masks'].dtype}")
    print(f"hidden shape: {text_feat['hidden'].shape}, type: {text_feat['hidden'].dtype}")
    print(f"position_ids shape: {text_feat['position_ids'].shape}, type: {text_feat['position_ids'].dtype}")
    print(f"text_token_mask shape: {text_feat['text_token_mask'].shape}, type: {text_feat['text_token_mask'].dtype}")

    visual_feats = model.extract_feat(img)

    if isinstance(visual_feats, (tuple, list)):
        for i, feat in enumerate(visual_feats):
            print(f"visual_feats[{i}] shape: {feat.shape}")
    else:
        print(f"visual_feats shape: {visual_feats.shape}")

    transformer_wrapper = TransformerWrapper(model, img_shape).eval()

    dynamic_axes = {
        "visual_feat_0": {0: "batch_size"},
        "visual_feat_1": {0: "batch_size"},
        "visual_feat_2": {0: "batch_size"},
        "visual_feat_3": {0: "batch_size"},
        "embedded": {0: "batch_size", 1: "seq_len"},
        "masks": {0: "batch_size", 1: "seq_len", 2: "seq_len_2"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            transformer_wrapper,
            args=(
                visual_feats[0],
                visual_feats[1],
                visual_feats[2],
                visual_feats[3],
                text_feat['embedded'],
                text_feat['masks'],
                text_feat['position_ids'],
                text_feat['text_token_mask'],
            ),
            f=output_path,
            input_names=[
                "visual_feat_0",
                "visual_feat_1",
                "visual_feat_2",
                "visual_feat_3",
                "embedded",
                "masks",
                "position_ids",
                "text_token_mask"
            ],
            output_names=["cls", "coords"],
            dynamic_axes=dynamic_axes,
            opset_version=16
        )
    print(f"[INFO] ONNX transformer model convert : {output_path}")

    ov_input = {"visual_feat_0":visual_feats[0],
                "visual_feat_1":visual_feats[1],
                "visual_feat_2":visual_feats[2],
                "visual_feat_3":visual_feats[3], 
                "embedded": text_feat['embedded'], 
                "masks": text_feat['masks'],
                "position_ids": text_feat['position_ids'], 
                "text_token_mask": text_feat['text_token_mask']}
    ov_save_path = output_path.parent.parent / 'IR_model' / output_path.name.replace('.onnx', '.xml')
    ov_save_path.parent.mkdir(parents=True, exist_ok=True)
    # ov_model = ov.convert_model(str(output_path), example_input=ov_input)
    ov_model = ov.convert_model(transformer_wrapper, example_input=ov_input)
    ov.save_model(ov_model, str(ov_save_path))
    print(f"[INFO] OpenVINO transformer model convert : {ov_save_path}")

def main():
    args = parse_args()
    init_args = {
        'model': args.conf,
        'weights': args.weights,
        'device': torch.device('cpu'),
        'palette': 'none',
        'show_progress': False,
    }
    # Init GroundingDINO-Swin-B model
    infer = DetInferencer(**init_args)
    model = infer.model.eval()
    img_shape = (int(args.h), int(args.w))
    cfg = model.language_model_cfg
    bert_path = Path(__file__).resolve().parent / 'models' / 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    script_dir = Path(__file__).resolve().parent
    base_name = f"gdino_swinb_{img_shape[0]}_{img_shape[1]}"
    onnx_dir = script_dir / 'models' / 'ONNX_model'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] OpenVINO Model Convert Start ...")
    
    export_language_model(model, tokenizer, cfg, onnx_dir / f"{base_name}_language.onnx")
    print("[INFO] OpenVINO Language Model Convert Done")
    
    export_visual_model(model, img_shape, onnx_dir / f"{base_name}_visual.onnx")
    print("[INFO] OpenVINO Vision Model Convert Done")
    
    export_transformer(model, img_shape, onnx_dir / f"{base_name}_transformer.onnx")
    print("[INFO] OpenVINO Transformer Model Convert Done")

    print("[INFO] OpenVINO Model Convert Done")


if __name__ == '__main__':
    main() 
