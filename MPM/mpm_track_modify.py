from mpm_track_parts_modify import trackp
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import os
import colorsys
import random
from glob import glob
import torch


def read_coords_from_pet(file_path):
    """
    从txt文件读取坐标（格式：id x y），只提取x, y
    """
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return coords
# 对于两种结果加权融合,小于阈值的按同一种处理,其中只有一方有添加在一起
def fuse_detections_weighted(det1, det2, threshold=5, weight1=0.7, weight2=0.3):
    """
    加权融合两个检测器的中心点坐标，返回格式为 [[id, x, y], ...]
    """
    det1 = np.array(det1)
    det2 = np.array(det2)

    if len(det1) == 0:
        return [[i, *coord] for i, coord in enumerate(det2)]
    if len(det2) == 0:
        return [[i, *coord] for i, coord in enumerate(det1)]

    dists = cdist(det1, det2)
    matched_1 = set()
    matched_2 = set()
    fused = []

    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            if dists[i, j] < threshold and i not in matched_1 and j not in matched_2:
                x = weight1 * det1[i][0] + weight2 * det2[j][0]
                y = weight1 * det1[i][1] + weight2 * det2[j][1]
                fused.append([x, y])
                matched_1.add(i)
                matched_2.add(j)

    for i in range(len(det1)):
        if i not in matched_1:
            fused.append(det1[i].tolist())
    for j in range(len(det2)):
        if j not in matched_2:
            fused.append(det2[j].tolist())

    fused_with_id = [[i, coord[0], coord[1]] for i, coord in enumerate(fused)]
    return fused_with_id


def mpm_track(seq_dir,save_dir,model,tkr,sigma):
     #对应的 原始图像位置
    image_folder = os.path.join(seq_dir, "origin")  
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files.sort() #使升序排列，防止文件夹中乱序，导致输入乱了
    image_paths = [os.path.join(image_folder, f) for f in files]


    
    # track_tab: Tracking records. shape is (, 11)
    # 0:frame, 1:id, 2:x, 3:y, 4:warped_x, 5:warped_y, 6:peak_value, 7:warped_pv, 8:climbed_wpv, 9:dis, 10:pid
    track_tab = []
    fused_pos = []
    seq_name = os.path.basename(seq_dir)
    # PET文件所在目录 pet定位
    pet_origin_dir = "/home/data_SSD/zk/pet_outputs/special_case"
    pet_dir = os.path.join(pet_origin_dir, seq_name,"txt")   #这里根据序列对应 指示对应的pet的坐标

     # 保存一些数据，什么数据待定
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 加载mpm模型,这里的size可以根据图片尺寸动态调整
    
    # Frame2 - -------------------------------------------------------------------
    for frame in range(0, len(image_paths)-1):
        print(f'-------{frame +1} - {frame + 2}-------')
        # association ------------------------------------------------------------
        mpm = tkr.inferenceMPM([image_paths[frame], image_paths[frame + 1]], model)
        mag = gaussian_filter(np.sqrt(np.sum(np.square(mpm), axis=-1)), sigma=sigma, mode='constant')
        
        mpm_pos = tkr.getMPMpoints(mpm, mag)
        
        image_name = os.path.basename(image_paths[frame])
        image_prefix = os.path.splitext(image_name)[0]
        # 动态拼接PET路径
        pet_path = os.path.join(pet_dir, f"{image_prefix}_pred_nms_positions.txt")
        pet_pos = read_coords_from_pet(pet_path)

        # 这里可以可以处理原始坐标,与pet像相结合   先按照顺序改吧,感觉逆序有点麻烦,先搞定一个版本
        fused_pos = fuse_detections_weighted(mpm_pos, pet_pos, threshold=8, weight1=0.6, weight2=0.4)

        # 保存 fused_pos 到每帧一个 txt 文件
        fused_path = os.path.join(save_dir, f"{image_prefix}_fused_positions.txt")
        with open(fused_path, "w") as f:
            for pos in fused_pos:
                f.write(f"{pos[0]},{pos[1]:.2f},{pos[2]:.2f}\n")
         
        # 运动匹配比对的是,预测predict_pos[]  与 当前检测的fused_pos
        # motion_map(fused_pos, pre_pos)  这里先只存储txt便于后面操作
    # 对于最后一帧特殊处理
    print('------- last_num ---> last_num - 1-------')
    last_num = len(image_paths) - 1 
    last_mpm = tkr.inferenceMPM([image_paths[last_num], image_paths[last_num-1]], model)
    last_mag = gaussian_filter(np.sqrt(np.sum(np.square(last_mpm), axis=-1)), sigma=sigma, mode='constant')
    mpm_pos = tkr.getMPMpoints(last_mpm, last_mag)
    
    image_name = os.path.basename(image_paths[last_num])
    image_prefix = os.path.splitext(image_name)[0]
    # save_dir = "/home/zk/diffusion-master/valid/mpm"
    # pet_path = 'valid/mpm/test_pet.txt'  #暂时先这样,后面可以动态获取
    # 动态拼接PET路径
    pet_path = os.path.join(pet_dir, f"{image_prefix}_pred_nms_positions.txt")
    pet_pos = read_coords_from_pet(pet_path)
    # 这里可以可以处理原始坐标,与pet像相结合   先按照顺序改吧,感觉逆序有点麻烦,先搞定一个版本
    fused_pos = fuse_detections_weighted(mpm_pos, pet_pos, threshold=8, weight1=0.6, weight2=0.4)
    # 保存 fused_pos 到每帧一个 txt 文件
    fused_path = os.path.join(save_dir, f"{image_prefix}_fused_positions.txt")
    with open(fused_path, "w") as f:
        for pos in fused_pos:
            f.write(f"{pos[0]},{pos[1]:.2f},{pos[2]:.2f}\n")
    with open("record.txt", "w+") as f:
        f.write(f"the num of sequence id{image_prefix},{len(fused_pos)}\n")


if __name__ == '__main__':
   
    model_path='model_best.pth'
    tkr = trackp(mag_th=0.3,itp=5,sigma=3,maxv=255,image_size=(1080, 1920))  
    # tracker
    # 加载预训练的MPMNet模型。如果parallel参数为True，则会在多GPU上并行加载模型。
    model = tkr.loadModel(model_path)

    root_dir = "/home/data_SSD/zk/dataset/special_case"
    # 遍历所有视频序列目录
    sequence_dirs = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        output_dir = os.path.join(root_dir, seq_name,"fused_txt")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n==> Start processing sequence: {seq_name}")
        try:
            mpm_track(seq_dir = seq_dir,save_dir= output_dir, model = model,tkr = tkr,sigma = 3)
        except Exception as e:
            print(f"[ERROR] Failed to process {seq_name}: {e}")
        torch.cuda.empty_cache()  # 清理显存
    
   


 




    