import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, pred, vis_dir, img_path, split_map=None):
    """
    Visualize predictions
    """
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw predictions (green)
        size = 3
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = img_path.split('/')[-1].split('.')[0]
            img_save_path = os.path.join(vis_dir, '{}_pred{}.jpg'.format(name, len(pred[idx])))
            cv2.imwrite(img_save_path, sample_vis)
            print('image save to ', img_save_path)

def visualization1(samples, pred, vis_dir, img_path, split_map=None):
    """
    Visualize predictions
    """

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # record [p_id,x,y] for save txt file
        person_id = 0
        size = 3
        pred_positions = []  # Store prediction positions for saving to txt file
        # draw predictions (green)
        for p in pred[idx]:
            # sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
            # pred_positions.append([person_id, int(p[1]), int(p[0])])
            # person_id +=1
            # 2024.11.21改：
            # if len(p) >= 2:  # Ensure there are at least two elements for indexing
            #     sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
            #     pred_positions.append([person_id, int(p[1]), int(p[0])])
            #     person_id += 1
            # 还是报错，2024.11.22改：
            if isinstance(p, torch.Tensor) and p.dim() >= 1 and len(p) >= 2:
                sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
                pred_positions.append([person_id, int(p[1]), int(p[0])])
                person_id += 1

        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis

        # save image
        if vis_dir is not None:
            # Create two folders to facilitate partition management of the two outputs

            # save visualization image
            vis_img_root = os.path.join(vis_dir, 'vis_img')
            if not os.path.exists(vis_img_root):
                os.makedirs(vis_img_root)

            # save individual coordinate
            txt_root = os.path.join(vis_dir, 'txt')
            if not os.path.exists(txt_root):
                os.makedirs(txt_root)

            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h + 1, :valid_w + 1]

            name = img_path.split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_img_root, '{}_pred_nms{}.jpg'.format(name, len(pred[idx]))),
                        sample_vis)

            # save predictions to txt file
            txt_path = os.path.join(txt_root, '{}_pred_nms_positions.txt'.format(name))
            with open(txt_path, 'w') as f:
                for person in pred_positions:
                    f.write('{},{},{}'.format(person[0], person[1], person[2]))
                    f.write('\n')

def nms(points, scores, dist_thresh):
    # 确定当前设备（CPU或GPU）
    device = points.device if isinstance(points, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保点的坐标和得分是张量并移动到相同的设备
    points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores = scores.clone().detach().to(dtype=torch.float32, device=device)

    # 检查输入的维度
    assert points.ndim == 2 and points.size(1) == 2, "points must be a tensor of shape [N, 2]"
    assert scores.ndim == 1, "scores must be a 1D tensor with one score per point"

    # 检查点的数量
    if points.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    # 按置信度排序点的索引
    sorted_indices = torch.argsort(scores, descending=True).to(device)

    # 初始化一个列表来保存最终保留下来的点的索引
    keep_indices = []

    while sorted_indices.numel() > 0:
        # 选择当前置信度最高的点
        current_index = sorted_indices[0]
        keep_indices.append(current_index)

        # 获取当前保留点的坐标
        current_point = points[current_index]

        # 获取剩余点并计算距离
        remaining_points = points[sorted_indices]
        if remaining_points.shape[0] > 1:
            distances = torch.cdist(current_point.view(1, -1), remaining_points).squeeze(0)
        else:
            distances = torch.tensor([0.0], device=device)

        # 找到距离大于阈值的点的索引
        mask = distances > dist_thresh
        sorted_indices = sorted_indices[mask]

    # 返回保留下来的点的坐标
    return points[torch.tensor(keep_indices, device=device, dtype=torch.long)]

@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir=None):
    model.eval()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # load image
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # transform image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    # inference
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    print('prediction: ', len(outputs_scores))
    
    # visualize predictions
    if vis_dir: 
        points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
        # 保存 points 到 .txt 文件
        dist_thresh = 20.0  #优化输出，除去置信度低的预测点
        kept_points = nms(points, outputs_scores, dist_thresh)

        # points_filename = os.path.join(vis_dir, 'predicted_points.txt')  此处的保存坐标点已经在visualization1中实现了
        # with open(points_filename, 'w') as f:                        
        #     for point in points:
        #         f.write(f"{point[0]}\t{point[1]}\n")  # 用制表符分隔点的 x, y 坐标
        # 这里还使用到numpy，必须要转到CPU上进行处理......
        split_map = (outputs['split_map_raw'][0].detach().squeeze(0) > 0.5).float().cpu().numpy()
        # 2024.11.21改：
        # visualization1(samples, targets, [kept_points], vis_dir, split_map=split_map)  
        visualization1(samples, [kept_points], vis_dir, img_path, split_map=split_map)

def main(args):
    # input image and model
    args.img_path = 'example/image/img110167.jpg'
    args.resume = 'best_checkpoint.pth'
    args.vis_dir = 'outputs/test_2'

    # build model
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)

    # load pretrained model
    checkpoint = torch.load(args.resume, map_location='cuda')        
    model.load_state_dict(checkpoint['model'])
    
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    evaluate_single_image(model, args.img_path, device, vis_dir=vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
