import os
from PIL import Image
from pathlib import Path  # 导入 pathlib 库

# 读取txt文件，获取坐标数据
def read_coordinates(txt_file):
    coordinates = []
    with open(txt_file, 'r') as file:
        for line in file:
            count, x, y = map(int, line.strip().split(','))
            coordinates.append((count, x, y))
    return coordinates

# 裁剪图片并保存
def crop_and_save_image(image_path, coordinates, output_folder, image_prefix):
    """裁剪图片，并按照 image_prefix 归类到同一个文件夹"""
    # 打开图片
    img = Image.open(image_path)

    # 创建前缀对应的子文件夹（如果不存在）
    image_subfolder = os.path.join(output_folder, image_prefix)
    os.makedirs(image_subfolder, exist_ok=True)

    # 获取图片名称（去除扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 遍历坐标列表，裁剪并保存图片
    for count, x, y in coordinates:
        left = x - 10  # 计算裁剪框的左边界，20x20的框中心是 x, y
        top = y - 10   # 计算裁剪框的上边界   为了超分 改成64*64
        right = x + 10 # 计算裁剪框的右边界
        bottom = y + 10 # 计算裁剪框的下边界

        # 裁剪图片
        cropped_img = img.crop((left, top, right, bottom))

        # 构造保存路径，文件名格式为：`原文件名_count_x_y.png`
        output_image_name = f"{image_name}_{count}_{x}_{y}.png"
        output_image_path = os.path.join(image_subfolder, output_image_name)

        # 使用 pathlib 统一文件路径格式，确保路径分隔符正确
        output_image_path = Path(output_image_path).as_posix()

        # 保存裁剪后的图片
        cropped_img.save(output_image_path)
        print(f"Saved cropped image: {output_image_path}")

# 处理文件夹中的所有图片和坐标文件
def process_folder(image_folder, txt_folder, output_folder):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 遍历每个图片文件
    for image_file in image_files:
        # 获取图片的前缀部分（如 img110167.jpg -> img110167）
        image_prefix = image_file.split(".")[0]

        # 查找对应的 txt 文件
        txt_file = os.path.join(txt_folder, f"{image_prefix}_pred_nms_positions.txt")
        print(txt_file)
        if os.path.exists(txt_file):
            print(f"Processing {image_file}...")
            coordinates = read_coordinates(txt_file)
            image_path = os.path.join(image_folder, image_file)
            crop_and_save_image(image_path, coordinates, output_folder, image_prefix)
        else:
            print(f"Warning: No matching txt file for {image_file}")

# 示例调用
image_folder = '/home/data_HDD/zk/dataset/00011/origin'  # 输入图片文件夹路径
txt_folder = '/home/data_HDD/zk/pet_outputs/00011/txt'  # 输入坐标数据txt文件夹路径
output_folder = '/home/data_HDD/zk/dataset/00011/test_sub_img'  # 输出文件夹路径

process_folder(image_folder, txt_folder, output_folder)