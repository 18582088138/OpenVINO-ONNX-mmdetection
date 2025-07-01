import os
import requests
from transformers import AutoTokenizer, AutoModel

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 定义模型名称和保存目录
model_name = "google-bert/bert-base-uncased"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "models", "bert-base-uncased")

# 检查目录是否存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"目录 {save_dir} 已创建。")

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"[INFO] 文件已存在，跳过下载: {save_path}")
        return
    print(f"[INFO] 正在下载: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] 下载完成: {save_path}")

# 下载 BERT 模型和tokenizer
try:
    print(f"正在从 HF-mirror 下载 {model_name} 到 {save_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"模型和tokenizer已成功下载并保存到 {save_dir}")
except Exception as e:
    print(f"下载失败: {e}")

# 下载 Grounding DINO 模型
model_url = "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"
model_save_path = os.path.join(script_dir, "models", "grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth")
download_file(model_url, model_save_path) 