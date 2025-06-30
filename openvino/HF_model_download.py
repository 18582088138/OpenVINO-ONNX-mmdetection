import os
import requests
from transformers import AutoTokenizer, AutoModel

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 定义模型名称和保存目录
model_name = "google-bert/bert-base-uncased"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "bert-base-uncased")

# 检查目录是否存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"目录 {save_dir} 已创建。")

def download_file(url, save_path):
    """
    下载文件并保存到指定路径。
    如果文件已存在，则跳过下载。
    """
    if os.path.exists(save_path):
        print(f"文件 {save_path} 已存在，跳过下载。")
        return

    try:
        print(f"正在从 {url} 下载文件...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求失败，则抛出异常

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件已成功保存到 {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")

# 下载模型和tokenizer
try:
    print(f"正在从 HF-mirror 下载 {model_name} 到 {save_dir} ...")
    
    # 只下载必要的文件，不使用cache
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    print(f"模型和tokenizer已成功下载并保存到 {save_dir}")

except Exception as e:
    print(f"下载失败: {e}")

# 下载 Grounding DINO 模型
grounding_dino_url = "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"
grounding_dino_save_path = os.path.join(script_dir, "grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth")

download_file(grounding_dino_url, grounding_dino_save_path) 