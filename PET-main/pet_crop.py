import os
from PIL import Image

# 读取txt文件，获取坐标数据
def read_coordinates(txt_file):
    coordinates = []
    with open(txt_file, 'r') as file:
        for line in file:
            count, x, y = map(int, line.strip().split(','))
            coordinates.append((count, x, y))
    return coordinates

# 裁剪图片并保存
def crop_and_save_image(image_path, coordinates, output_folder):
    # 打开图片
    img = Image.open(image_path)
    
    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取图片名称（去除扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 遍历坐标列表，裁剪并保存图片
    for count, x, y in coordinates:
        left = x - 10  # 计算裁剪框的左边界，20x20的框中心是x, y
        top = y - 10   # 计算裁剪框的上边界
        right = x + 10 # 计算裁剪框的右边界
        bottom = y + 10 # 计算裁剪框的下边界
        
        # 裁剪图片
        cropped_img = img.crop((left, top, right, bottom))
        
        # 构造保存路径，图片名称为原文件名 + count
        output_image_name = f"{image_name}_{count}.png"
        output_image_path = os.path.join(output_folder, output_image_name)
        
        # 保存裁剪后的图片
        cropped_img.save(output_image_path)
        print(f"Saved cropped image: {output_image_path}")

# 处理文件夹中的所有图片和坐标文件
def process_folder(image_folder, txt_folder, output_folder):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 遍历每个图片文件
    for image_file in image_files:
        # 获取图片文件的前缀
        image_prefix = os.path.splitext(image_file)[0]
        
        # 查找对应的txt文件
        txt_file = os.path.join(txt_folder, f"{image_prefix}_pred_nms_positions.txt")
        if os.path.exists(txt_file):
            print(f"Processing {image_file}...")
            coordinates = read_coordinates(txt_file)
            image_path = os.path.join(image_folder, image_file)
            crop_and_save_image(image_path, coordinates, output_folder)
        else:
            print(f"Warning: No matching txt file for {image_file}")

# 示例调用
image_folder = 'example/image/test_1'  # 输入图片文件夹路径
txt_folder = 'outputs/test_2/txt'  # 输入坐标数据txt文件夹路径
output_folder = '/home/zk/LAVIS-main/test_sub_img'  # 输出文件夹路径

process_folder(image_folder, txt_folder, output_folder)