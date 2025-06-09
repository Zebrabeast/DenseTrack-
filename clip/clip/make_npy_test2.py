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
"""针对test_1的改进，1.进行了边界判断，超出边界的点，设置裁剪范围处于图像内
                    2.从原来的循环，裁剪一个，提取一个改为裁剪一帧，提取一帧，后面可能还要改成数据集  """
# def process_and_extract(image_folder, txt_folder, feature_save_root):
#     os.makedirs(feature_save_root, exist_ok=True)

#     image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

#     for image_file in image_files:
#         image_prefix = image_file.split(".")[0]
#         txt_file = os.path.join(txt_folder, f"{image_prefix}_fused_positions.txt")
#         if not os.path.exists(txt_file):
#             print(f"❌ 找不到匹配的坐标文件: {txt_file}")
#             continue

#         image_path = os.path.join(image_folder, image_file)
#         img = Image.open(image_path).convert("RGB")
#         coordinates = read_coordinates(txt_file)  # [(pid, x, y), ...]

#         if not coordinates:
#             continue

#         crops, pid_xy_list = [], []
#         for pid, x, y in coordinates:
#             W, H = img.size  # 注意 PIL 图像是 (宽, 高)  控制裁剪的范围，使其在之中
#             left = max(0, x - 10)
#             top = max(0, y - 10)
#             right = min(W, x + 10)
#             bottom = min(H, y + 10)
#             cropped_img = img.crop((left, top, right, bottom))
#             crops.append(cropped_img)  #这里的对应，保证数据是对应的，后面如果有问题，可以从这里查看
#             pid_xy_list.append((pid, x, y))

#         # 批量图像处理
#         image_tensors = torch.stack([vis_processors["eval"](c) for c in crops]).to(device)
#         text_input = txt_processors["eval"]("A pedestrian")
#         sample = {"image": image_tensors, "text_input": [text_input] * len(crops)}
#         features = model.extract_features(sample)
#         image_embeds = features.image_embeds.cpu().detach().numpy()

#         frame_data = []
#         for (pid, x, y), embed in zip(pid_xy_list, image_embeds):
#             frame_data.append((pid, (x, y), embed))
#             print(f"✅ 提取特征: {image_prefix}_{pid}_{x}_{y} | 维度: {embed.shape}")

#         if frame_data:
#             structured_data = np.array(frame_data, dtype=dtype)
#             save_path = os.path.join(feature_save_root, f"{image_prefix}.npy")
#             np.save(save_path, structured_data)
#             print(f"💾 已保存特征: {save_path}")

#     print("✅ 全部处理完成！")
# def process_and_extract(image_folder, txt_folder, feature_save_root):
#     os.makedirs(feature_save_root, exist_ok=True)

#     image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

#     for image_file in image_files:
#         image_prefix = image_file.split(".")[0]
#         txt_file = os.path.join(txt_folder, f"{image_prefix}_fused_positions.txt")
#         if not os.path.exists(txt_file):
#             print(f"❌ 找不到匹配的坐标文件: {txt_file}")
#             continue

#         image_path = os.path.join(image_folder, image_file)
#         img = Image.open(image_path).convert("RGB")
#         coordinates = read_coordinates(txt_file)  # [(pid, x, y), ...]

#         if not coordinates:
#             continue

#         crops, pid_xy_list = [], []
#         for pid, x, y in coordinates:
#             W, H = img.size  # 注意 PIL 图像是 (宽, 高)  控制裁剪的范围，使其在之中
#             left = max(0, x - 10)
#             top = max(0, y - 10)
#             right = min(W, x + 10)
#             bottom = min(H, y + 10)
#             cropped_img = img.crop((left, top, right, bottom))
#             crops.append(cropped_img)  #这里的对应，保证数据是对应的，后面如果有问题，可以从这里查看
#             pid_xy_list.append((pid, x, y))
#         frame_data = []
#         # 批量图像处理 不知道有没有区别哈
#         for (pid, x, y), cropped_img in zip(pid_xy_list, crops):
#             image_tensor = vis_processors["eval"](cropped_img).unsqueeze(0).to(device)
#             text_input = txt_processors["eval"]("A pedestrian")
#             sample = {"image": image_tensor, "text_input": [text_input]}
#             feature_image = model.extract_features(sample)
#             image_embed = feature_image.image_embeds.squeeze(0).cpu().detach().numpy()

#            #保存提取的特征，由于可能的显存超出问题，暂时这么做
#             frame_data.append((pid, (x, y), image_embed))
#             print(f"✅ 提取特征: {image_prefix}_{pid}_{x}_{y} | 维度: {image_embed.shape}")

#         if frame_data:
#             structured_data = np.array(frame_data, dtype=dtype)
#             save_path = os.path.join(feature_save_root, f"{image_prefix}.npy")
#             np.save(save_path, structured_data)
#             print(f"💾 已保存特征: {save_path}")

#     print("✅ 全部处理完成！")
   #再次改进，使用batch_size提升速度，鉴于显存占用少，GPU占用也少

def process_and_extract(image_folder, txt_folder, feature_save_root, batch_size= 16):
    os.makedirs(feature_save_root, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_prefix = image_file.split(".")[0]
        txt_file = os.path.join(txt_folder, f"{image_prefix}_fused_positions.txt")
        #这里是针对已存在的过滤，便于终端恢复
        save_path = os.path.join(feature_save_root, f"{image_prefix}.npy")
        # ✅ 检查是否已经存在提取好的特征
        if os.path.exists(save_path):
            print(f"⏭️ 已存在特征文件，跳过: {save_path}")
            continue
        
        if not os.path.exists(txt_file):
            print(f"❌ 找不到匹配的坐标文件: {txt_file}")
            continue

        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path).convert("RGB")
        coordinates = read_coordinates(txt_file)  # [(pid, x, y), ...]

        if not coordinates:
            continue

        # 收集裁剪图像和对应的 pid, x, y
        crops, pid_xy_list = [], []
        for pid, x, y in coordinates:
            W, H = img.size  # 注意 PIL 图像是 (宽, 高)
            left = max(0, x - 10)
            top = max(0, y - 10)
            right = min(W, x + 10)
            bottom = min(H, y + 10)
            cropped_img = img.crop((left, top, right, bottom))
            crops.append(cropped_img)
            pid_xy_list.append((pid, x, y))

        # 批量处理图像
        frame_data = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_pid_xy_list = pid_xy_list[i:i + batch_size]

            # 转换为 tensor 并移动到设备
            image_tensors = torch.stack([vis_processors["eval"](c) for c in batch_crops]).to(device)
            text_input = txt_processors["eval"]("A pedestrian")
            sample = {"image": image_tensors, "text_input": [text_input] * len(batch_crops)}

            features = model.extract_features(sample)
            image_embeds = features.image_embeds.cpu().detach().numpy()

            for (pid, x, y), embed in zip(batch_pid_xy_list, image_embeds):
                frame_data.append((pid, (x, y), embed))
                print(f"✅ 提取特征: {image_prefix}_{pid}_{x}_{y} | 维度: {embed.shape}")

        # 保存为 npy 文件
        if frame_data:
            structured_data = np.array(frame_data, dtype=dtype)
            save_path = os.path.join(feature_save_root, f"{image_prefix}.npy")
            np.save(save_path, structured_data)
            print(f"💾 已保存特征: {save_path}")

    print("✅ 全部处理完成！")


# #  原图地址
# image_folder = '/home/data_SSD/zk/dataset/00011/origin'
# # 融合坐标地址
# txt_folder = '/home/data_SSD/zk/dataset/00011/fused_txt'

# feature_save_root = '/home/data_SSD/zk/dataset/00011/features'

# process_and_extract(image_folder, txt_folder, feature_save_root)



root_dir = '/home/data_SSD/zk/dataset/special_case'

sequence_dirs = sorted([
    os.path.join(root_dir, d) for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])
#这里到00110序列号
for seq_dir in sequence_dirs:
    seq_name = os.path.basename(seq_dir)
    txt_folder =  os.path.join(seq_dir,"fused_txt")
    feature_save_root = os.path.join(seq_dir, "features") #输出 特征进行保存
    os.makedirs(feature_save_root, exist_ok=True)
    #对应的 原始图像位置
    image_folder = os.path.join(seq_dir, "origin")
    print(f"\n==> Start processing sequence: {seq_name}")
    try:
        process_and_extract(image_folder, txt_folder, feature_save_root)
    except Exception as e:
        print(f"[ERROR] Failed to process {seq_name}: {e}")
    torch.cuda.empty_cache()  # 清理显存