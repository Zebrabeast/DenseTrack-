import os
import re
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载CLIP模型
model, vis_processors, txt_processors = load_model_and_preprocess(
    "clip_feature_extractor", model_type="ViT-B-16", is_eval=True, device=device
)

# 保存特征的结构定义
dtype = np.dtype([
    ('id', np.int32),         # 行人ID
    ('coordinate', np.float32, 2),  # 中心点坐标(x, y)
    ('feature', np.float32, 512)    # CLIP输出特征
])

# 读取 txt 中的坐标
def read_coordinates(txt_file):
    coordinates = []
    with open(txt_file, 'r') as file:
        for line in file:
            count_str, x_str, y_str = line.strip().split(',')
            count = int(count_str)
            x = float(x_str)
            y = float(y_str)
            coordinates.append((count, x, y))
    return coordinates

# 主处理函数：裁剪 + 提特征 + 保存
"""针对test的改进，1.将裁剪部分集成到提取部分了，省了保存裁剪图像的步骤，
                     暂时设定为裁出一张检查一张"""
def process_and_extract(image_folder, txt_folder, feature_save_root):
    os.makedirs(feature_save_root, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_prefix = image_file.split(".")[0]
        # if image_prefix != "img110300":   之前漏下的部分（当时为了检查输出是否合理）
        #     continue
        txt_file = os.path.join(txt_folder, f"{image_prefix}_fused_positions.txt")
        if not os.path.exists(txt_file):
            print(f"❌ 找不到匹配的坐标文件: {txt_file}")
            continue

        # 读取图像和坐标
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path).convert("RGB")
        coordinates = read_coordinates(txt_file)

        frame_data = []

        for pid, x, y in coordinates:
            left, top, right, bottom = x - 10, y - 10, x + 10, y + 10
            cropped_img = img.crop((left, top, right, bottom))

            # 特征提取
            image_tensor = vis_processors["eval"](cropped_img).unsqueeze(0).to(device)
            text_input = txt_processors["eval"]("A pedestrian")
            sample = {"image": image_tensor, "text_input": [text_input]}
            feature_image = model.extract_features(sample)
            image_embed = feature_image.image_embeds.squeeze(0).cpu().detach().numpy()

            # 存储为结构化数据
            frame_data.append((pid, (x, y), image_embed))
            print(f"✅ 提取特征: {image_prefix}_{pid}_{x}_{y} | 维度: {image_embed.shape}")

        # 保存为npy
        if frame_data:
            structured_data = np.array(frame_data, dtype=dtype)
            save_path = os.path.join(feature_save_root, f"{image_prefix}.npy")
            np.save(save_path, structured_data)
            print(f"💾 已保存特征: {save_path}")

    print("✅ 全部处理完成！")

#  原图地址
image_folder = '/home/data_HDD/zk/dataset/00011/origin'
# 融合坐标地址
txt_folder = '/home/data_HDD/zk/dataset/00011/fused_txt'

feature_save_root = '/home/data_HDD/zk/dataset/00011/features'

process_and_extract(image_folder, txt_folder, feature_save_root)