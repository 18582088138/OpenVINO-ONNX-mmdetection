import torch
from argparse import ArgumentParser
from pathlib import Path
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from transformers import AutoTokenizer
from mmdet.models.language_models.bert import generate_masks_with_special_tokens_and_transfer_map

import openvino as ov
core = ov.Core()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default="../configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py", help='Config file')
    parser.add_argument('--weights', type=str, default="./models/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth", help='*.pth file')
    parser.add_argument('--h', type=str, default="800", help='input width')
    parser.add_argument('--w', type=str, default="1333", help='input height')

    args = parser.parse_args()
    return args

class GroundingDINOWrapper(torch.nn.Module):
    def __init__(self, model, img_shape):
        super().__init__()
        self.model = model
        self.img_meta = dict(
            img_shape=img_shape,
            pad_shape=img_shape,
            batch_input_shape=img_shape,
        )
    def forward(self, img, input_ids, position_ids, text_token_mask):
        text_inputs = {
            'input_ids': input_ids,
            'attention_mask': text_token_mask,
            'position_ids': position_ids,
            'token_type_ids': torch.zeros_like(input_ids),
        }
        text_feat = self.model.language_model.language_backbone(text_inputs)
        text_feat['embedded'] = self.model.text_feat_map(text_feat['embedded'])
        text_feat['position_ids'] = position_ids
        text_feat['text_token_mask'] = torch.ones_like(input_ids).bool()
        bs = img.shape[0]
        batched_data_samples = []
        for i in range(bs):
            data_sample = DetDataSample(metainfo=self.img_meta)
            batched_data_samples.append(data_sample)

        visual_feats = self.model.extract_feat(img)
        head_inputs_dict = self.model.forward_transformer(visual_feats, text_feat, batched_data_samples)
        outs = self.model.bbox_head(
            memory_text=head_inputs_dict["memory_text"],
            text_token_mask=head_inputs_dict["text_token_mask"],
            hidden_states=head_inputs_dict["hidden_states"],
            references=head_inputs_dict["references"])
        return outs[0][-1], outs[1][-1]


def main():
    args = parse_args()
    init_args = {
        'model': args.conf,
        'weights': args.weights,
        'device': torch.device('cpu'),
        'palette': 'none',
        'show_progress': False,
    }

    infer = DetInferencer(**init_args)

    img_shape = (int(args.h), int(args.w))
    batch_inputs = torch.randn(1,3,*(img_shape))

    classes = ('Face', 'Person', 'Pet', 'Vehicle', 'Plate', 'Nonmotor', 'Head')
    text_prompts = [' . '.join(classes) + ' .']

    cfg = infer.model.language_model_cfg
    tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parent / 'models' / 'bert-base-uncased')
    tokenized = tokenizer.batch_encode_plus(
                text_prompts,
                max_length=cfg.max_tokens,
                padding='max_length' if cfg.pad_to_max else 'longest',
                return_special_tokens_mask=True,
                return_tensors='pt',
                truncation=True)

    input_ids = tokenized['input_ids']
    input_names = ["img", "input_ids", "position_ids", "text_token_mask"]
    output_names = ["cls", "coords"]

    text_token_mask, position_ids = \
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, infer.model.language_model.special_tokens)

    dynamic_axes={
        "img": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
        "cls": {0: "batch_size"},
        "coords": {0: "batch_size"},
    }

    model_wrapper = GroundingDINOWrapper(infer.model.eval(), img_shape).eval()

    script_dir = Path(__file__).resolve().parent
    onnx_model_path = script_dir / f"./models/ONNX_model/gdino_swinb_{img_shape[0]}_{img_shape[1]}.onnx"
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model_wrapper,
            args=(batch_inputs, input_ids, position_ids, text_token_mask),
            f=str(onnx_model_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=16)

    print(f"[INFO] ONNX model saved as {onnx_model_path}")

    ov_model_path = script_dir / f"./models/IR_model/gdino_swinb_{img_shape[0]}_{img_shape[1]}.xml"
    ov_model_path.parent.mkdir(parents=True, exist_ok=True)
    ov_input = {"img":batch_inputs,
                "input_ids":input_ids,
                "position_ids":position_ids,
                "text_token_mask":text_token_mask}
    ov_model = ov.convert_model(
        str(onnx_model_path),
        example_input=ov_input,
    )
    ov.save_model(ov_model, ov_model_path)
    print(f"[INFO] OpenVINO model saved as {ov_model_path}")


if __name__ == '__main__':
    main()