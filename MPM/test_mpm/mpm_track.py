from mpm_track_parts import trackp
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os
import colorsys
import random
from glob import glob


# import matplotlib
# matplotlib.use('tkagg')

def track(image_names, model_path, save_dir, mag_th=0.3, itp=5, sigma=3, maxv=255):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tkr = trackp(mag_th, itp, sigma, maxv, (1040, 1392))  # tracker
    # 加载预训练的MPMNet模型。如果parallel参数为True，则会在多GPU上并行加载模型。
    model = tkr.loadModel(model_path)

    # 颜色生成：生成一系列随机颜色，用于可视化跟踪结果。后面似乎没有用
    ID_COLOR = [colorsys.hsv_to_rgb(h, 0.8, 0.8) for h in np.linspace(0, 1, 100)]
    random.shuffle(ID_COLOR)
    ID_COLOR = np.array(ID_COLOR)

    # track_tab: Tracking records. shape is (, 11)
    # 0:frame, 1:id, 2:x, 3:y, 4:warped_x, 5:warped_y, 6:peak_value, 7:warped_pv, 8:climbed_wpv, 9:dis, 10:pid
    track_tab = []
    not_ass_tab = np.empty((0, 11))  # Not association table
    pop_tab = np.empty((0, 11))  # Appear table
    div_tab = np.empty((0, 11))  # Division table

    # Frame0 - farme1
    print('-------0 - 1-------')
    # 这一步返回的应该是第一帧第二帧之间的pre_mpm图
    pre_mpm = tkr.inferenceMPM([image_names[0], image_names[1]], model)
    pre_mag = gaussian_filter(np.sqrt(np.sum(np.square(pre_mpm), axis=-1)), sigma=sigma, mode='constant')
    pre_pos = tkr.getMPMpoints(pre_mpm, pre_mag)
    for i, each_pos in enumerate(pre_pos):
        # register frame(0 & 1)
        track_tab.append([0, i + 1] + each_pos[2:4] + [0, 0, 0, 0, 0, 0, 0])
        track_tab.append([1, i + 1] + each_pos + [pre_mag[int(each_pos[1]), int(each_pos[0])], 0, 0, 0])
    track_tab = np.array(track_tab)
    # saveResultImage(1, image_names[0], image_names[1], pre_mag, np.zeros(pre_mag.shape),
    #                 track_tab[track_tab[:, 0] == 1], div_tab)
    pre_pos = track_tab[track_tab[:, 0] == 1]
    new_id = len(pre_pos) + 1

    # with open("track_tab.txt", "w") as f:
    #     f.write("frame\tid\tx\ty\twarped_x\twarped_y\tpeak_value\twarped_pv\tclimbed_wpv\tdis\tpid\n")
    #     for row in track_tab:
    #         f.write(f"{int(row[0])}\t{int(row[1])}\t" + "\t".join(f"{x}" for x in row[2:]) + "\n")
    
    with open("track_tab_div_1.txt", "w") as f:
        for row in pre_pos:
            f.write(f"{int(row[0])},{int(row[1])}," + ",".join(f"{x}" for x in row[2:6]) + "\n")

    # Frame2 - -------------------------------------------------------------------
    for frame in range(2, len(image_names)):
        print(f'-------{frame - 1} - {frame}-------')
        # association ------------------------------------------------------------
        mpm = tkr.inferenceMPM([image_names[frame - 1], image_names[frame]], model)
        mag = gaussian_filter(np.sqrt(np.sum(np.square(mpm), axis=-1)), sigma=sigma, mode='constant')
        pos = tkr.getMPMpoints(mpm, mag)
        add_ass, add_not_ass, add_div, add_pop, new_id = tkr.associateCells(frame, pos, pre_pos, pre_mag, new_id)
        div_tab = np.append(div_tab, add_div, axis=0)
        if len(pop_tab) > 0:
            pop_tab = np.append(pop_tab, add_pop, axis=0)
        not_ass_tab = np.append(not_ass_tab, add_not_ass, axis=0)
        # saveResultImage(frame, image_names[frame - 1], image_names[frame], mag, pre_mag, ass_tab, add_div_tab)
        pre_mag = mag.copy()
        pre_pos = add_ass.copy()
        track_tab = np.append(track_tab, add_ass, axis=0)

    # tkr.saveCTK(track_tab, not_ass_tab, div_tab, save_dir)
    return


if __name__ == '__main__':
    demo = track(
        image_names=['img110001.jpg', 'img110006.jpg'],
        model_path='model_best.pth',
        save_dir='test_mpm')
