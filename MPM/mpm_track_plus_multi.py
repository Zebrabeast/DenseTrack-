from mpm_track_parts_plus import trackp
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import os
import colorsys
import random
from glob import glob
import torch

import time
import argparse
import pickle
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion

#因为需要调用global  故需要最顶层
Candidate_pool = {}    # 全局缓冲池，跨帧累积
can_traj = 0           # 全局轨迹ID递增器
can_len = 0

class Trajectory:
    def __init__(self, obj_id, feature, coordinate, max_len=5, decay=0.87):
        self.obj_id = obj_id
        self.features = [feature]  # 存储历史特征
        self.current_coordinate = coordinate  # 当前坐标
        self.predicted_coordinate = None  # 初始化预测坐标
        self.max_len = max_len  # 特征历史长度
        self.decay = decay  # 权重衰减系数
        self.unmatched_act = 0

    # def update(self, new_feature, new_coordinate):
    #     """更新轨迹信息"""
    #     self.features.append(new_feature)
    #     if len(self.features) > self.max_len:
    #         self.features.pop(0)

    #     self.current_coordinate = new_coordinate  # 更新当前坐标
    #     # 预测坐标可以在外部调用 set_prediction() 来更新
    
    def update(self, new_feature, new_coordinate=None):
        if new_feature is not None:
            self.features.append(new_feature)
            if len(self.features) > self.max_len:
                self.features.pop(0)
            self.unmatched_act = 0  # 匹配成功，清零
        else:
            self.unmatched_act += 1  # 未匹配，加一

        if new_coordinate is not None:
            self.current_coordinate = new_coordinate

    def predict(self, mpm):
        pos = self.current_coordinate
        # 这里要求索引为整数，故需要更改pos,将其强转为整数型
        pos_x = int(round(pos[0]))
        pos_y = int(round(pos[1]))
        
        # 边界检查
        height, width = 1080,1920  # 假设 mpm 是 NumPy 数组，形状 (1080, 1920)
        pos_x = max(0, min(pos_x, width - 1))   # 限制在 [0, 1919] 范围内
        pos_y = max(0, min(pos_y, height - 1))  # 限制在 [0, 1079] 范围内
        
        vec = mpm[pos_y, pos_x]
        # mag_value = pre_mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        if abs(vec[2]) < 0.9:
            vec[2] = 0.98
        x = 10000 if vec[2] == 0 else int(5 * (vec[1] / vec[2]))  # Calculate the motion vector x
        y = 10000 if vec[2] == 0 else int(5 * (vec[0] / vec[2]))  # Calculate the motion vector y
        pred_coord = [pos[0] + x, pos[1] + y]  # mag_value,暂时还不知道有什么用
        # """设置预测坐标"""  经过检查发现是一个特殊情况，加个判定值，这里基本vec[2] 为0.9几，
        # with open("vec_log.txt", "a") as f:  # 使用追加模式写入
        #    f.write(f"pos=({pos_x},{pos_y}),偏移量={x,y} vec={vec.tolist()}\n")

        self.predicted_coordinate = pred_coord

    def get_weighted_feature(self):
        """计算加权平均特征"""
        weights = np.array([self.decay ** i for i in range(len(self.features))][::-1])  # 递减权重
        weights /= np.sum(weights)  # 归一化
        return np.sum(np.array(self.features) * weights[:, None], axis=0)

    def get_position(self):
        """返回当前和预测坐标"""
        return {
            "current": self.current_coordinate,
            "predicted": self.predicted_coordinate
        }
 
class Candidate_traj:
    def __init__(self, obj_id, feature, frame_id, coordinate, max_len=10, decay=0.87):
        self.obj_id = obj_id
        self.history = []
        self.hits = 1      # 匹配命中次数    根据命中次数可以进行添加进主轨迹中
        self.age = 1       # 总存在帧数     防止过多累计，将其中的长时间未匹配到的进行清除可以设置为 20 
        self.max_len = max_len  #这里设置
        self.decay = decay
        self.last_update_frame = frame_id
        self.current_coordinate = coordinate
        self.predicted_coordinate = None
        self.history.append({   #初始化时调用这个，只添加这一个
            'frame': frame_id,
            'coord': coordinate,
            'feature': feature
        })

    def add_record(self, frame_id, feature, coordinate):
        self.history.append({
            'frame': frame_id,
            'coord': coordinate,
            'feature': feature
        })
        self.hits += 1
        self.age += 1
        self.current_coordinate = coordinate
        self.last_update_frame = frame_id
        # if len(self.history) > self.max_len:
        #     self.history.pop(0)
    def get_weighted_feature(self): #用于对比的缓冲池特征
        """计算加权平均特征"""
        features = [x['feature'] for x in self.history[-5:]]
        weights = np.array([self.decay ** i for i in range(len(features))][::-1])  # 递减权重
        weights /= np.sum(weights)  # 归一化
        return np.sum(np.array(features) * weights[:, None], axis=0)

    def predict(self, mpm): #用于对比的预测坐标
        pos = self.current_coordinate
        # 这里要求索引为整数，故需要更改pos,将其强转为整数型
        pos_x = int(round(pos[0]))
        pos_y = int(round(pos[1]))
        # 边界检查
        height, width = 1080,1920 # 假设 mpm 是 NumPy 数组，形状 (1080, 1920)
        pos_x = max(0, min(pos_x, width - 1))   # 限制在 [0, 1919] 范围内
        pos_y = max(0, min(pos_y, height - 1))  # 限制在 [0, 1079] 范围内
        
        vec = mpm[pos_y, pos_x]
        # mag_value = pre_mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        if abs(vec[2]) < 0.9:
            vec[2] = 0.98
        x = 10000 if vec[2] == 0 else int(5 * (vec[1] / vec[2]))  # Calculate the motion vector x
        y = 10000 if vec[2] == 0 else int(5 * (vec[0] / vec[2]))  # Calculate the motion vector y
        pred_coord = [pos[0] + x, pos[1] + y]  # mag_value,暂时还不知道有什么用
        # """设置预测坐标"""  经过检查发现是一个特殊情况，加个判定值，这里基本vec[2] 为0.9几，
        # with open("vec_log.txt", "a") as f:  # 使用追加模式写入
        #    f.write(f"pos=({pos_x},{pos_y}),偏移量={x,y} vec={vec.tolist()}\n")

        self.predicted_coordinate = pred_coord

def clean_candidate_pool(candidate_pool, current_frame,
                         max_inactive_frames= 20, 
                         min_confirm_hits=13,
                         age_threshold=16):
    to_remove = []   #20   8   12   一个组合，其实还好,但是增加量有点太多了额额额 25 17 20增加量还行，不过存在着一个问题，似乎第一次匹配的结果再逐步变小
    #这里删除长时间不更新的候选轨迹、未被确定存在时间长的，和已经被确定要移出去的
    for obj_id, traj in candidate_pool.items():
        inactive_time = current_frame - traj.last_update_frame
        if inactive_time > max_inactive_frames:
            to_remove.append(obj_id)
        elif traj.hits < min_confirm_hits and traj.age > age_threshold:
            to_remove.append(obj_id)
        elif traj.hits >= min_confirm_hits:
            to_remove.append(obj_id)
    print("已经删除了的候选池的数量",len(to_remove))
    for obj_id in to_remove:
        del candidate_pool[obj_id]

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

def reset_globals():
    global Candidate_pool, can_traj,can_len 
    Candidate_pool = {}
    can_traj = 0           # 全局轨迹ID递增器
    can_len = 0

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


def mpm_track(seq_dir, save_dir, model, tkr,sigma=3):

     #对应的 原始图像位置
    image_folder = os.path.join(seq_dir, "origin")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files.sort() #使升序排列，防止文件夹中乱序，导致输入乱了
    image_paths = [os.path.join(image_folder, f) for f in files]
   
    seq_name = os.path.basename(seq_dir)
    # PET文件所在目录 pet定位
    pet_origin_dir = "/home/data_SSD/zk/pet_outputs/special_case"
    pet_dir = os.path.join(pet_origin_dir, seq_name,"txt")   #这里根据序列对应 指示对应的pet的坐标

    features_dir = os.path.join(seq_dir, "features")
    # track_tab: Tracking records. shape is (, 11)
    # 0:frame, 1:id, 2:x, 3:y, 4:warped_x, 5:warped_y, 6:peak_value, 7:warped_pv, 8:climbed_wpv, 9:dis, 10:pid
    res = []
    fused_pos = []

    traj_id = 0
    trajectories = {}  # 需要确保为 字典类型

    # Frame1 - ----------------------------------- 从第一帧向后，不过最后一帧需要特殊处理
    for frame in range(0, len(image_paths)):  # 直接到最后一帧300进行特殊处理
        
        image_name = os.path.basename(image_paths[frame])
        image_prefix = os.path.splitext(image_name)[0]
        img_npy = os.path.join(features_dir, f"{image_prefix}.npy")
        query_data = np.load(img_npy, allow_pickle=True)
        seq_id = image_prefix[-3:]
        print(f"Processing: {img_npy}, Sequence ID: {seq_id}")

        if frame != len(image_paths) - 1:
            print(f'-------{frame + 1} - {frame + 2}-------')
            # association ------------------------------------------------------------
            mpm = tkr.inferenceMPM([image_paths[frame], image_paths[frame + 1]], model)
            mag = gaussian_filter(np.sqrt(np.sum(np.square(mpm), axis=-1)), sigma=sigma, mode='constant')
            mpm_pos = tkr.getMPMpoints(mpm, mag)

            # # 动态拼接PET路径   为了方便看效果，这里的坐标我已经提前提取好，用于提取特征了
            # pet_path = os.path.join(pet_dir, f"{image_prefix}_pred_nms_positions.txt")
            # pet_pos = read_coords_from_pet(pet_path)

            # # 这里可以可以处理原始坐标,与pet像相结合   先按照顺序改吧,感觉逆序有点麻烦,先搞定一个版本
            # fused_pos = fuse_detections_weighted(mpm_pos, pet_pos, threshold=8, weight1=0.6, weight2=0.4)
            # # 保存 fused_pos 到每帧一个 txt 文件
            # fused_path = os.path.join(save_dir, f"{image_prefix}_fused_positions.txt")
            # with open(fused_path, "w") as f:
            #     for pos in fused_pos:
            #         f.write(f"{pos[0]},{pos[1]:.2f},{pos[2]:.2f}\n")

            # 对于第一帧进行初始化轨迹操作 frame1 ------------------------------------------
            if frame == 0:
                for i in range(len(query_data["coordinate"])):
                    # 初始化轨迹
                    obj_id = query_data["id"][i]
                    coord = query_data["coordinate"][i]  # 当前目标中心坐标 [x, y]
                    feat = query_data["feature"][i]
                    # 初始化 Trajectory 实例
                    traj = Trajectory(
                        obj_id=traj_id,
                        feature=feat,
                        coordinate=coord
                    )
                    trajectories[traj_id] = traj
                    traj_id += 1
                    # 创建 sub_res（一般是用于可视化或结果写入）
                    sub_res = [1, obj_id, round(coord[0] - 10,2), round(coord[1] - 10,2), 20, 20,
                               1, 1, -1, -1]
                    res.append(sub_res)
                # 更新预测的坐标
                for traj in trajectories.values():
                    traj.predict(mpm)
                # 初始化后继续下一帧
                global can_traj, Candidate_pool
                continue
                
        # Frame2 - ------------------------------------------------------------------
        # 运动匹配比对的是,预测predict_pos[]  与 当前检测的fused_pos
        # motion_map(fused_pos, pre_pos)
        # 经过多重匹配得来的ass_tab
        else:
            print('------- last_num ---> last_num - 1-------')
            # 由于这个是n到n+1，计算当前位置时，最后一帧处理不到，因此要特殊处理
            # 这一步返回的应该是第一帧第二帧之间的pre_mpm图
            last_num = len(image_paths) - 1
            last_mpm = tkr.inferenceMPM([image_paths[last_num], image_paths[last_num - 1]], model)
            last_mag = gaussian_filter(np.sqrt(np.sum(np.square(last_mpm), axis=-1)), sigma=sigma, mode='constant')
            mpm_pos = tkr.getMPMpoints(last_mpm, last_mag)

            image_name = os.path.basename(image_paths[last_num])
            image_prefix = os.path.splitext(image_name)[0]
            mag = last_mag  # 这里统一一下变量，方便传入数据
            mpm = last_mpm
            # 同样这里的坐标处理先给注释掉
            # pet_path = os.path.join(pet_dir, f"{image_prefix}_pred_nms_positions.txt")
            # pet_pos = read_coords_from_pet(pet_path)
            # # 这里可以可以处理原始坐标,与pet像相结合   先按照顺序改吧,感觉逆序有点麻烦,先搞定一个版本
            # fused_pos = fuse_detections_weighted(mpm_pos, pet_pos, threshold=8, weight1=0.6, weight2=0.4)
            # # 保存 fused_pos 到每帧一个 txt 文件
            # fused_path = os.path.join(save_dir, f"{image_prefix}_fused_positions.txt")
            # with open(fused_path, "w") as f:
            #     for pos in fused_pos:
            #         f.write(f"{pos[0]},{pos[1]:.2f},{pos[2]:.2f}\n")

        matched_one, matched_second,unmatched_trks, better_unmatched_dets = \
            assign_cascade(trks=trajectories, dets=query_data, tkr=tkr, mag=mag,image_prefix =image_prefix,mpm = mpm)
        # 匹配成功的部分
        for m in matched_one:
            trajectories[m[1]].update(query_data['feature'][m[0]], query_data['coordinate'][m[0]])
            sub_res = [int(image_prefix[-3:]), m[1],
                       round(query_data['coordinate'][m[0]][0] - 10,2), round(query_data['coordinate'][m[0]][1] - 10,2),
                       20, 20, 1, 1, -1, -1]
            res.append(sub_res)
        for m in matched_second:
            trajectories[m[1]].update(query_data['feature'][m[0]], query_data['coordinate'][m[0]])
            sub_res = [int(image_prefix[-3:]), m[1],
                        round(query_data['coordinate'][m[0]][0] - 10,2), round(query_data['coordinate'][m[0]][1] - 10,2),
                       20, 20, 1, 1, -1, -1]
            res.append(sub_res)

        # 没匹配到的什么轨迹，后面可以用判断边界，设置出局什么的
        for m in unmatched_trks:
            trajectories[m].update(None)

        # #经过缓冲池过滤过的 unmatched_dets,
        for traj in better_unmatched_dets.values():
            first = True
            for record in traj.history:
                if first:
                    new_traj_id = traj_id
                    trajectories[new_traj_id] = Trajectory(new_traj_id, record['feature'], record['coord'])
                    first = False
                    traj_id += 1
                else:
                    trajectories[new_traj_id].update(record['feature'], record['coord'])

                sub_res = [record['frame'], new_traj_id,
                        round(record['coord'][0] - 10, 2), round(record['coord'][1] - 10, 2),
                        20, 20, 1, 1, -1, -1]
                res.append(sub_res)
        # 更新预测的坐标
        for traj in trajectories.values():
            traj.predict(mpm)
    # 保存结果
    seq_head = image_prefix[-6:-3]
    file_path =  os.path.join(save_dir, f"00{seq_head}_diffusion_clip_dis.txt")
    with open(file_path, "w+") as f:
        for item in res:
            f.write(','.join(map(str, item)) + '\n')

    #最后重置候选池，避免不同视频序列之间的干扰。
    reset_globals()


# 两者都重复，均可取，只有单方的话，可信度不是那么高了吧  一对多什么的，有必要考虑？
def min_cost_matching(motion_pre_assign, appearance_pre_assign):
    # 并在一起   #  如果只有三个参数，插入到最后一列，变成四列，  整体的预处理
    if motion_pre_assign[0].shape[0]:
        if appearance_pre_assign[0].shape[0]:
            pre_matches = np.vstack((np.insert(motion_pre_assign[0], 3, 0, axis=1), np.insert(
                appearance_pre_assign[0], 3, 1, axis=1)))
        # 由于这里单个的并不可靠，所以只采取交集作为匹配成功的
        # 未匹配的并在一起 这里需要改动未匹配的内容需要加上一下内容，比如并在一起

        unmatched_dets = list(set(motion_pre_assign[1]).union(
            set(appearance_pre_assign[1])))
        unmatched_tracks = list(set(motion_pre_assign[2]).union(
            set(appearance_pre_assign[2])))
        print("第一次匹配(#`O′)未处理前---------")
        print("unmatched_detections", len(unmatched_dets))
        print("unmatched_tracks", len(unmatched_tracks))

    if pre_matches.shape[0]:
        # 找出冲突   感觉这里不是很合理啊，将某一边单独检测到的添加到match中，处理冲突，最终只采用外观很像的，大约0.9
        #     这里的双方都匹配到的，可以进行添加到高置信度，但是单边的可信度不太高，然后  对于后面的处理，不知是否有意义
        # 后面的 two_round的匹配，是对于某 非常高的匹配值添加为某个  ，不太清楚  ，但是感觉并不一定能用
        matches, conflicts = split_TD_conflict(pre_matches)
    print(" the first high matching num:", len(matches), "---------------")
    # 计算差集 det  
    pre_match_det_ids = set([int(match[0]) for match in pre_matches])  # 提取 det_idx
    match_det_ids = set([int(match[0]) for match in matches])
    # 计算差集 trk   这里由于前面记录了scores，且后面又np.array 所以这里需要换成int,不然后面会报错
    pre_match_trk_ids = set([int(match[1])for match in pre_matches])  # 提取 trk_idx
    match_trk_ids = set([int(match[1]) for match in matches])
    # 计算差集并合并到 unmatched_dets 和 unmatched_tracks
    unmatched_dets = list(set(unmatched_dets).union(pre_match_det_ids - match_det_ids))
    unmatched_tracks = list(set(unmatched_tracks).union(pre_match_trk_ids - match_trk_ids))
    print("第一次匹配(#`O′)已处理后---------")
    print("unmatched_detections", len(unmatched_dets))
    print("unmatched_tracks", len(unmatched_tracks))
    matches = sorted(matches, key=lambda m: m[0])
    # 等会仔细看在哪加
    matches = [(int(m[0]), int(m[1]), m[2]) for m in matches]
    print("排序后的matches（前5个）:", matches[:5])
    return matches, unmatched_tracks, unmatched_dets

# 冲突匹配，额 消减不正常的匹配
def split_TD_conflict(pre_matches):
    """
     保留两个模型都匹配到相同 (track_id, det_id) 的情况
     即：track_id 出现两次，并且它的 det_id 是相同的（表示两个模型都给出相同结果）
     """
    T_vals, T_indexes, T_counts = better_np_unique(pre_matches[:, 0])
    matches, matches_ind = [], []

    # single_idxs = np.where(T_counts == 1)[0]
    # for i in single_idxs:
    #     T_ind = int(T_indexes[i])
    #     dv = pre_matches[T_ind, 1]
    #     c = D_counts[np.argwhere(D_vals == dv)]
    #     if c == 1:
    #
    #         matches.append(pre_matches[T_ind, :])
    #
    #         matches_ind.append(T_ind)

    two_idxs = np.where(T_counts == 2)[0]
    for i in two_idxs:
        T_ind0 = int(T_indexes[i][0])
        T_ind1 = int(T_indexes[i][1])
        dv0 = pre_matches[T_ind0, 1]
        dv1 = pre_matches[T_ind1, 1]
        if dv0 == dv1:
            matches.append(pre_matches[T_ind0, :])
            matches_ind.append(T_ind0)
            matches_ind.append(T_ind1)

    conflicts_ind = list(
        set(np.arange(pre_matches.shape[0])) - set(matches_ind))
    conflict_matches = pre_matches[conflicts_ind, :]
    # 结果并不干净，比如 confilct_results可能包含着重复的内容
    # 只保留前三个元素 即为 det,trk,score
    matches = [m[:3] for m in matches]
    return list(map(tuple, matches)), conflict_matches

def better_np_unique(arr):
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, _, counts = np.unique(arr,
                                               return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, counts

    # #这个没有用到，感觉不是很符合这个
    # def two_round_match(conflicts, alpha_gate):

    #     if conflicts.shape[0] == 0:
    #         return [], [], []

    #     first_round = conflicts[conflicts[:, 2] >= alpha_gate, :]
    #     matches_a = first_round[first_round[:, 3] == 1, :]
    #     if len(matches_a) != 0:
    #         matches_a[:, 4] = 2

    #     second_round = conflicts[conflicts[:, 2] < alpha_gate, :]
    #     second_round = second_round[second_round[:, 3] == 0, :]
    #     if len(second_round) != 0:
    #         second_round[:, 4] = 3
    #     second_round = select_pairs(second_round, 0, matches_a[:, 0])

    #     second_round = select_pairs(second_round, 1, matches_a[:, 1])

    #     matches_b = second_round[:, :]

    #     matches_ = np.vstack((matches_a, matches_b))
    #     matches_ = matches_.astype(np.int_)
    #     unmatched_tracks_ = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    #     unmatched_tracks_ = [int(i) for i in unmatched_tracks_]
    #     unmatched_detections_ = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    #     unmatched_detections_ = [int(i) for i in unmatched_detections_]

    #     return list(map(tuple, matches_)), unmatched_tracks_, unmatched_detections_
    # def select_pairs(matches, col_id, pairs):

    for t in pairs:
        ind = np.argwhere(matches[:, col_id] == t)
        if len(ind):
            matches[ind[0], 3] = -1
    matches = matches[matches[:, 3] == 0, :]

    return matches

# 特征怎么对齐呢,和坐标  # 这里使用KM,后面又怎么处理
def apperance_associate(det_one, trks, detection_indices, track_indices, motion_scores):
    # 调用函数，获得分数  #！！！这里的track_indices是存在隐患的，但是由于轨迹只会后面添加，不会中间替换删除，故可以这么做
    scores = get_appreance_scores(det_one,trks,motion_scores)
    cost_matrix = - scores
    pre_matches = []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 存在着部分问题，比如不知道索引的含义等等等  track_indices
    for row, col in zip(row_ind, col_ind):
        # 这里暂时保留，防止后面有什么改的东西    这里的detection_idx可用可不用，不会错，传进去的索引
        detection_idx = detection_indices[row]
        track_idx = track_indices[col]
        # detection_idx = row
        # track_idx = col

        if scores[row, col] > 0.75:
            pre_matches.append((detection_idx, track_idx, scores[row, col]))

    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_detections = list(set(detection_indices) - set(list(pre_matches[:, 0])))
        unmatched_tracks = list(set(track_indices) - set(list(pre_matches[:, 1])))
        # 这里添加列向量似乎与后面有些重复，这里的匹配内容我只返回三列
        # pre_matches = np.hstack(
        #     (pre_matches, np.ones(pre_matches.shape[0]).reshape(-1, 1)))
    else:
        unmatched_detections = detection_indices
        unmatched_tracks = track_indices

    # 待修改，我先往后顺，看这个索引有什么用
    return (pre_matches, unmatched_detections, unmatched_tracks), abs(scores.T)

def level_match(track_indices, detection_one_indices,trks, dets_one, depth, tkr,mag):
    # if track_indices is None:
    #     track_indices = np.arange(len(trks))
    if detection_one_indices is None:
        detection_one_indices = np.arange(len(dets_one))
   


    print("---------------第一次匹配----------------")
    if len(detection_one_indices) != 0 and len(track_indices) != 0:
        # 这里使用的是最小化匹配，它使用的是0.3    那么相似度应该大于0.7
        motion_pre_assign, motion_scores = tkr.motion_associate(dets_one, trks,track_indices,mag, distance_threshold=10)
        appearance_pre_assign, emb_cost = apperance_associate(dets_one, trks, detection_one_indices, track_indices,
                                                              motion_scores)

        matched_one, unmatched_trks_1, unmatched_dets_one = min_cost_matching(motion_pre_assign,
                                                                                  appearance_pre_assign)

    # 二段匹配，从其中的unmatched_trk其中降低标准，再次匹配  例如距离什么的，继续匹配
    unmatched_trks_one = unmatched_trks_1
    
    print("---------------第二次匹配----------------")
    matched_second, unmatched_trks_second, unmatched_dets_second = second_associate(trks, dets_one, unmatched_trks_one,
                                                                                    unmatched_dets_one, tkr, mag)

    unmatched_trks_2 = unmatched_trks_second
    unmatched_dets_2 = unmatched_dets_second
    # #  三次匹配，只要不是他离谱都可以  可以再降一下，比如距离小于多少，外观只要不是小于0.4还是0.5
    # if unmatched_dets_2.shape[0] > 0 and unmatched_trks_2.shape[0] > 0:
    #     print("---------------第三次匹配----------------")
    #     matched_third, unmatched_trks_third, unmatched_dets_third =  third_associate(trks, dets_one,
    #             unmatched_trks_2, unmatched_dets_2,tkr, mag)
    # else:
    #     matched_one_2 = np.empty((0, 2), dtype=int)
    #     unmatched_trks_third = unmatched_trks_2
    #     unmatched_dets_third = unmatched_dets_2
  
    return matched_one, matched_second,unmatched_trks_2, unmatched_dets_2

# assign_cascade   进行匹配，可以将匹配结果在这里面添加到轨迹中，更好看
def assign_cascade(trks, dets, tkr, mag,image_prefix,mpm):
    track_indices = list(trks.keys())
    detection_one_indices = list(range(len(dets["id"])))
    # 这个索引是用于子集 映射回原始位置用的，这里可能暂时没用到，不影响
    unmatched_dets_one = detection_one_indices
    dets_one = dets
    unmatched_trks_l = []

    for depth in range(1):

        if len(unmatched_dets_one) == 0 :
            break
        #  这里是级联匹配的处理方式，感觉可要可不要  不确定要不要加上所谓的长时间的匹配
        # track_indices_l = [
        #     k for k in track_indices if self.trackers[k].time_since_update >= 1 + depth]

        # if len(track_indices_l) == 0:
        #     continue
        #主流匹配过程
        matched_one, matched_second,unmatched_trks, unmatched_dets = \
            level_match(track_indices, detection_one_indices, trks, dets_one,
                        depth, tkr, mag)

        # 正好在这里进行 unmatched_dets的缓冲池再过滤操作
        global Candidate_pool, can_traj
        frame_id = int(image_prefix[-3:])
        # unmatched_dets_dict  =  dets_one[unmatched_dets]
        better_unmatched_dets = {}
        can_matched = []
        unmatched_trks_can = []
        unmatched_dets_can = []
        if(len(unmatched_dets)!=0):#优化处理，对于没有未匹配到的不做处理
            if can_traj == 0:  #当缓冲池里没有的时候
                for i in unmatched_dets:
                    Candidate_pool[can_traj] = Candidate_traj(can_traj, dets['feature'][i],frame_id,dets['coordinate'][i])  #这里记录特征，坐标，和帧数
                    Candidate_pool[can_traj].predict(mpm) #更新预测坐标方便比对
                    can_traj += 1
            else:
                #则进行匹配处理
                can_matched,unmatched_trks_can,unmatched_dets_can = candidate_associate(Candidate_pool,dets_one,unmatched_dets,tkr,mag)   #暂定这样
                #针对于匹配结果的处理
                for m in can_matched: #对于匹配到的，更新记录，并且添加内容，并且更新预测值
                    Candidate_pool[m[1]].add_record(frame_id,dets['feature'][m[0]], dets['coordinate'][m[0]])
                    Candidate_pool[m[1]].predict(mpm)
                for m in unmatched_trks_can:#对于候选池未匹配到的，更新存在的age
                    Candidate_pool[m].age += 1

                for m in unmatched_dets_can: #对于二次匹配没匹配到，候选池也没匹配到的，加入候选池中
                
                    Candidate_pool[can_traj] = Candidate_traj(can_traj, dets['feature'][m],
                                                            frame_id,dets['coordinate'][m])  #这里记录特征，坐标，和帧数
                    Candidate_pool[can_traj].predict(mpm) #更新预测坐标方便比对
                    can_traj += 1
                 
        
        #然后判断缓冲池中是否又满足某些内容得值，提取出来返回  出来得为better_unmatched，直接添加到主轨迹中
        # 提拔高质量候选轨迹    
        better_unmatched_dets = {i: traj for i, traj in enumerate(Candidate_pool.values()) if traj.hits >= 13}
        print("Better unmatched:", len(better_unmatched_dets))
        #设置一个记录帧数得变量，定期清理缓冲池    
        clean_candidate_pool(Candidate_pool, frame_id)                       
    #
    # ret = self.final_process(matched_one)
    #
    # if len(ret) > 0:
    #     return np.concatenate(ret)
    return matched_one, matched_second,unmatched_trks,better_unmatched_dets

def fuse_appearance_motion_scores(appearance_scores, motion_scores, modified_scores, max_distance=15.0):
    """
    根据距离动态计算外观和动作的权重，并返回融合后的分数矩阵。
    """
    # 正规化 motion 分数: (max_distance - d) / max_distance
    # 反推 distance（来自 motion_score）
    distance_est = max_distance * (1 - motion_scores)
    # modified_matrix = np.sqrt(motion_scores) * 10
    # 将距离映射为外观权重：距离越小，alpha越大，范围 [0.3, 0.5]  虽然想用0.4到0.6的，但是外观不用占比很高
    alpha = 0.5 - ((distance_est / max_distance) * 0.2)
    appearance_weights = np.clip(alpha, 0.3, 0.5)
    motion_weights = 1.0 - appearance_weights  # 保证和为1

    # 融合分数矩阵（逐元素加权）
    fused_scores = np.multiply(appearance_weights, appearance_scores) + np.multiply(motion_weights, modified_scores)

    # 设置 gate：只保留 motion_scores > 某个阈值的（防止乱匹配）
    # 这个是防止，外观的乱匹配，将距离过远的设为0
    min_motion_score = 0.3  # 相当于 距离大于 10.5
    fused_scores[motion_scores < min_motion_score] = 0.0

    return fused_scores

def second_associate(trks, dets_one, unmatched_trks, unmatched_dets, tkr, mag):
    # 这是原始的索引
    # unmatched_trks = trks[unmatched_trks]  并不能直接这样使用
    unmatched_trks_dict = {tid: trks[tid] for tid in unmatched_trks}
    pred_pos = [traj.predicted_coordinate for traj in unmatched_trks_dict.values()]
    pos = dets_one["coordinate"]
    # 定义距离阈值
    # 二次匹配可以宽松点，适当扩大范围   比较纠结这个东西，按照实际来看移动的似乎很小吧
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=15)
    # 额，微调，扩大一下，感觉没啥用  反而更接近的感觉
    modified_scores = np.sqrt(motion_scores)
    appearance_scores = get_appreance_scores(dets_one,unmatched_trks_dict, modified_scores)
    # 获取两者加权融合的分数
    fused_scores = fuse_appearance_motion_scores(appearance_scores, motion_scores, modified_scores, max_distance=15)
    row_indices, col_indices = linear_sum_assignment(-fused_scores)  # 取负是因为匈牙利算法默认求最小值

    # 获取匹配结果   前面的过滤是将超过阈值的设为0，便于最大化匹配，但是部分并没有除去，还需要再判断
    matched_pairs = []
    unmatched_det_set = set(unmatched_dets)
    # 然后在判断里用：
    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = fused_scores[det_idx, trk_idx]
        if score > 0.65 and det_idx in unmatched_det_set:  # 只考虑有效匹配（分数大于）  分数大于暂定于0.5
            matched_pairs.append((det_idx, unmatched_trks[trk_idx], score))

    print(f'# 第二次匹配关联到的 associated people: {len(matched_pairs)}')

    # 提取已匹配的索引
    matched_det_indices = [m[0] for m in matched_pairs]
    matched_trk_indices = [m[1] for m in matched_pairs]
    # 计算未匹配索引
    unmatched_trks_2 = np.setdiff1d(unmatched_trks, matched_trk_indices)
    unmatched_dets_2 = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# 未匹配轨迹 unmatched_trk: {len(unmatched_trks_2)}')
    print(f'# 未匹配目标点 unmatched_det: {len(unmatched_dets_2)}')

    return matched_pairs, unmatched_trks_2, unmatched_dets_2

def second_associate_modified(trks, track_indices,dets_one, unmatched_trks, unmatched_dets, tkr, mag):
    # 这是原始的索引
    # unmatched_trks = trks[unmatched_trks]  并不能直接这样使用 这里与上一个不同的地方在于unmatched_dets->_trks 感觉差不多
    pred_pos = [traj.predicted_coordinate for traj in trks.values()]
    unmatched_dets_dict = dets_one[unmatched_dets]
    pos = unmatched_dets_dict["coordinate"]
    # 定义距离阈值
    # 二次匹配可以宽松点，适当扩大范围   比较纠结这个东西，按照实际来看移动的似乎很小吧
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=15)
    # 额，微调，扩大一下，感觉没啥用  反而更接近的感觉
    modified_scores = np.sqrt(motion_scores)
    appearance_scores = get_appreance_scores(unmatched_dets_dict,trks, modified_scores)
    # 获取两者加权融合的分数
    fused_scores = fuse_appearance_motion_scores(appearance_scores, motion_scores, modified_scores, max_distance=15)
    row_indices, col_indices = linear_sum_assignment(-fused_scores)  # 取负是因为匈牙利算法默认求最小值

    # 获取匹配结果   前面的过滤是将超过阈值的设为0，便于最大化匹配，但是部分并没有除去，还需要再判断
    matched_pairs = []
    unmatched_trk_set = set(unmatched_trks)
    # 然后在判断里用：
    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = fused_scores[det_idx, trk_idx]
        if score > 0.65 and track_indices[trk_idx] in unmatched_trk_set:  # 只考虑有效匹配（分数大于）  分数大于暂定于0.5
            matched_pairs.append((unmatched_dets[det_idx], track_indices[trk_idx], score))

    print(f'# 第二次匹配关联到的 associated people: {len(matched_pairs)}')

    # 提取已匹配的索引
    matched_det_indices = [m[0] for m in matched_pairs]
    matched_trk_indices = [m[1] for m in matched_pairs]
    # 计算未匹配索引
    unmatched_trks_2 = np.setdiff1d(unmatched_trks, matched_trk_indices)
    unmatched_dets_2 = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# 未匹配轨迹 unmatched_trk: {len(unmatched_trks_2)}')
    print(f'# 未匹配目标点 unmatched_det: {len(unmatched_dets_2)}')

    return matched_pairs, unmatched_trks_2, unmatched_dets_2

def third_associate(trks, dets_one, unmatched_trks, unmatched_dets, tkr, mag):
    # 这是原始的索引
    # unmatched_trks = trks[unmatched_trks]  并不能直接这样使用
    unmatched_trks_dict = {tid: trks[tid] for tid in unmatched_trks}
    pred_pos = [traj.predicted_coordinate for traj in unmatched_trks_dict.values()]
    unmatched_dets_dict = dets_one[unmatched_dets]
    pos = unmatched_dets_dict["coordinate"]
    # 定义距离阈值
    # 二次匹配可以宽松点，适当扩大范围   比较纠结这个东西，按照实际来看移动的似乎很小吧
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=15)
    # 额，微调，扩大一下，感觉没啥用  反而更接近的感觉
    modified_scores = np.sqrt(motion_scores)
    appearance_scores = get_appreance_scores(unmatched_dets_dict, unmatched_trks_dict, modified_scores,weight = 0.7)
    row_indices, col_indices = linear_sum_assignment(-appearance_scores)  # 取负是因为匈牙利算法默认求最小值

    # 获取匹配结果   前面的过滤是将超过阈值的设为0，便于最大化匹配，但是部分并没有除去，还需要再判断
    matched_pairs = []
    # 然后在判断里用：
    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = appearance_scores[det_idx, trk_idx]
        if score > 0.4:  # 只考虑有效匹配（分数大于）  分数大于暂定于0.4
            matched_pairs.append((unmatched_dets[det_idx], unmatched_trks[trk_idx], score))

    print(f'# 第三次匹配关联到的 associated people: {len(matched_pairs)}')

    # 提取已匹配的索引
    matched_det_indices = [m[0] for m in matched_pairs]
    matched_trk_indices = [m[1] for m in matched_pairs]
    # 计算未匹配索引
    unmatched_trks_3 = np.setdiff1d(unmatched_trks, matched_trk_indices)
    unmatched_dets_3 = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# 未匹配轨迹 unmatched_trk: {len(unmatched_trks_3)}')
    print(f'# 未匹配目标点 unmatched_det: {len(unmatched_dets_3)}')
    #只能说大致比较可靠
    # filename = f"test_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write("## 匹配的轨迹坐标：\n")
    #     for tid in matched_pairs:
    #         f.write(f"目标 ID {tid[0]} 坐标: {dets_one['coordinate'][tid[0]]} 轨迹 ID {tid[1]} 坐标: {trks[tid[1]].current_coordinate} 分数：{tid[2]}\n")
    #     f.write("## 未匹配轨迹坐标：\n")
    #     for tid in unmatched_trks_3:
    #         coord = trks[tid].predicted_coordinate
    #         f.write(f"轨迹 ID {tid} 坐标: {coord}\n")

    #     f.write("\n## 未匹配目标点坐标：\n")
    #     for did in unmatched_dets_3:
    #         coord = dets_one["coordinate"][did]
    #         f.write(f"目标 ID {did} 坐标: {coord}\n")

    return matched_pairs, unmatched_trks_3, unmatched_dets_3

def candidate_associate(candidate_pool, dets_one,unmatched_dets, tkr, mag):
    # 这是原始的索引
    # unmatched_trks = trks[unmatched_trks]  并不能直接这样使用
    candidate_keys = list(candidate_pool.keys())
    pred_pos = [traj.predicted_coordinate for traj in candidate_pool.values()] #这里得trks是cad_traj即为候选池
    unmatched_dets_dict = dets_one[unmatched_dets]
    pos = unmatched_dets_dict["coordinate"]
    # 定义距离阈值
    # 二次匹配可以宽松点，适当扩大范围   比较纠结这个东西，按照实际来看移动的似乎很小吧
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=15)
    # 额，微调，扩大一下，感觉没啥用  反而更接近的感觉
    modified_scores = np.sqrt(motion_scores)  #未检测点与候选轨迹池之间的比对
    appearance_scores = get_appreance_scores(unmatched_dets_dict, candidate_pool, modified_scores,weight = 0.4)
    row_indices, col_indices = linear_sum_assignment(-appearance_scores)  # 取负是因为匈牙利算法默认求最小值

    # 获取匹配结果   前面的过滤是将超过阈值的设为0，便于最大化匹配，但是部分并没有除去，还需要再判断
    matched_pairs = []
    # 然后在判断里用：
    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = appearance_scores[det_idx, trk_idx]
        if score > 0.8:  # 只考虑有效匹配（分数大于）  分数大于暂定于0.6
            matched_pairs.append((unmatched_dets[det_idx], candidate_keys[trk_idx], score))

    print(f'# 第三次匹配 通过候选池关联到的 associated people: {len(matched_pairs)}')

    # 提取已匹配的索引
    matched_det_indices = [m[0] for m in matched_pairs]
    matched_candidate_indices = [m[1] for m in matched_pairs]
    # 计算未匹配索引
    unmatched_trks_cand = np.setdiff1d(candidate_keys, matched_candidate_indices)
    unmatched_dets_cand = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# 未匹配轨迹 unmatched_trk in candidate_pool: {len(unmatched_trks_cand)}')
    print(f'# 未匹配目标点 unmatched_det: {len(unmatched_dets_cand)}')
    #只能说大致比较可靠
    # filename = f"test_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write("## 匹配的轨迹坐标：\n")
    #     for tid in matched_pairs:
    #         f.write(f"目标 ID {tid[0]} 坐标: {dets_one['coordinate'][tid[0]]} 轨迹 ID {tid[1]} 坐标: {trks[tid[1]].current_coordinate} 分数：{tid[2]}\n")
    #     f.write("## 未匹配轨迹坐标：\n")
    #     for tid in unmatched_trks_3:
    #         coord = trks[tid].predicted_coordinate
    #         f.write(f"轨迹 ID {tid} 坐标: {coord}\n")

    #     f.write("\n## 未匹配目标点坐标：\n")
    #     for did in unmatched_dets_3:
    #         coord = dets_one["coordinate"][did]
    #         f.write(f"目标 ID {did} 坐标: {coord}\n")

    return matched_pairs, unmatched_trks_cand, unmatched_dets_cand

def get_appreance_scores(dets_one,trks, motion_scores,weight = 0.6):
    """匹配 queries 到轨迹，通过改动这里可以决定是否使用diffusion什么的"""
    gallery_ids = list(trks.keys())
    gallery_features = np.array([trks[tid].get_weighted_feature() for tid in gallery_ids])
    query_features = dets_one["feature"]
    n_query = len(query_features)
    trun_size = max(n_query,len(gallery_ids))
    if trun_size > 100: #这里似乎只要满足比他们相加和小一点即可
        args.truncation_size = (trun_size // 100) * 100
    elif trun_size >= 10:  # 10 ≤ trun_size ≤ 100
        args.truncation_size = (trun_size // 10) * 10
    else:  # trun_size < 10
        args.truncation_size = trun_size  # 个位数原样保留
    diffusion = Diffusion(np.vstack([query_features, gallery_features]), args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    print(f"offline.shape: {offline.shape}")
    scores = features[:n_query] @ features[n_query:].T

    # normalized_queries = preprocessing.normalize(det_one["feature"], norm="l2", axis=1)
    # normalized_gallery = preprocessing.normalize(gallery_features, norm="l2", axis=1)
    # scores = normalized_queries @ normalized_gallery.T  # 这里是不经过diffusion处理的,直接计算的
    print("appearance_scores----------------------")
    # 计算距离矩阵（欧几里得距离）
    query_coordinates = np.array(dets_one["coordinate"])  # 假设query_data中包含坐标
    print("查询目标坐标数", query_coordinates.shape)
    gallery_coordinates = np.array([trks[tid].current_coordinate for tid in gallery_ids])
    print("轨迹的坐标数", gallery_coordinates.shape)
    distances = cdist(query_coordinates, gallery_coordinates, metric='euclidean')
    distance_threshold = 15
    scores[distances > distance_threshold] = float('0')

    # 这里给外观加点动作，希望后面匹配的多一点
    fused_scores = scores * weight + motion_scores * (1 - weight)

    return fused_scores

def residual_cosine_distance(x, y, temperature=100, data_is_normalized=False):
    """
    Args:
        x: np.ndarray, shape (num_tracklets, feature_dim)
        y: np.ndarray, shape (num_detections, feature_dim)
        temperature: scaling factor for softmax
        data_is_normalized: if True, skip feature normalization
    Returns:
        distances: np.ndarray, 1 - enhanced cosine similarity (越小越相似)
        similarities: np.ndarray, enhanced cosine similarity matrix
    """
    if not data_is_normalized:
        x = np.asarray(x)
        y = np.asarray(y)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)

    ftrk = torch.from_numpy(x).half().cuda()  # track
    fdet = torch.from_numpy(y).half().cuda()  # detection
    # 原始余弦相似度 (tracking → detection)
    aff = torch.mm(fdet, ftrk.transpose(0, 1))  # 调整顺序为 detection → track
    # 残差重建的softmax归一化
    aff_td = F.softmax(temperature * aff, dim=1)  # detection → track
    aff_dt = F.softmax(temperature * aff, dim=0).transpose(0, 1)  # track → detection
    # 残差重构
    res_recons_ftrk = torch.mm(aff_td, fdet)  # track 重构
    res_recons_fdet = torch.mm(aff_dt, ftrk)  # detection 重构
    # 融合后的相似度
    enhanced_sim = (torch.mm(fdet, ftrk.transpose(0, 1)) + torch.mm(res_recons_fdet, res_recons_ftrk.transpose(0, 1))) / 2

    return enhanced_sim.cpu().numpy()  #这里的返回值是什么


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--query_path',
                        type=str,
                        required=False,
                        help="""
                        Path to query features
                        """)
    parser.add_argument('--gallery_path',
                        type=str,
                        required=False,
                        help="""
                        Path to gallery features
                        """)
    parser.add_argument('--gnd_path',
                        type=str,
                        help="""
                        Path to ground-truth
                        """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=300,
                        help="""
                        Number of images in the truncated gallery
                        """)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args

if __name__ == '__main__':
    
    args = parse_args()
    model_path='model_best.pth'
    tkr = trackp(mag_th=0.3,itp=5,sigma=3,maxv=255,image_size=(1080, 1920))  
    # tracker
    # 加载预训练的MPMNet模型。如果parallel参数为True，则会在多GPU上并行加载模型。
    model = tkr.loadModel(model_path)
    over_seq = ["00102","00104","00106","00107","00109"]
    root_dir = "/home/data_SSD/zk/dataset/special_case"
    # 遍历所有视频序列目录
    sequence_dirs = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        # if seq_name not in over_seq:
        #     continue
        output_dir = os.path.join(root_dir, seq_name,"result_mod")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n==> Start processing sequence: {seq_name}")
        mpm_track(seq_dir = seq_dir,save_dir= output_dir, model = model,tkr = tkr,sigma = 3)
        # try:
        #     mpm_track(seq_dir = seq_dir,save_dir= output_dir, model = model,tkr = tkr,sigma = 3)
        # except Exception as e:
        #     print(f"[ERROR] Failed to process {seq_name}: {e}")
        torch.cuda.empty_cache()  # 清理显存
    