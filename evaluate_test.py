import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


@torch.inference_mode()
def evaluate_test(net, dataloader, device, amp, num_classes=4):
    net.eval()
    num_val_batches = len(dataloader)

    # 初始化混淆矩阵 (H, W) -> (真实类别, 预测类别)
    conf_matrix = torch.zeros((num_classes, num_classes), device=device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 预测并获取类别索引
            output = net(image)
            mask_pred = output.argmax(dim=1)

            # 更新混淆矩阵
            # 这里的计算会将所有像素累加到 num_classes x num_classes 的矩阵中
            conf_matrix += torch.bincount(
                num_classes * mask_true.reshape(-1) + mask_pred.reshape(-1),
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # 计算各项指标
    conf_matrix = conf_matrix.cpu().numpy()
    eps = 1e-7

    # 1. 总体精度 OA
    oa = np.diag(conf_matrix).sum() / (conf_matrix.sum() + eps)

    # 2. 计算每个类别的精度 (PA, UA, IoU)
    # PA (生产者精度/召回率): 矩阵行之和
    pa_list = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + eps)
    # UA (用户精度/准确率): 矩阵列之和
    ua_list = np.diag(conf_matrix) / (conf_matrix.sum(axis=0) + eps)
    # IoU
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou_list = intersection / (union + eps)
    # F1 Score (即每个类别的 Dice)
    f1_list = 2 * (pa_list * ua_list) / (pa_list + ua_list + eps)

    net.train()

    # 返回一个字典，包含所有你论文中需要的指标
    return {
        'OA': oa,
        'PA': pa_list,  # 这是一个数组，包含 [未扰动, 砍伐, 火灾, 非森林] 的 PA
        'UA': ua_list,
        'F1': f1_list,
        'IoU': iou_list,
        'mIoU': np.mean(iou_list),
        'Dice_Avg': np.mean(f1_list[1:])  # 通常遥感评估会关注扰动类（1,2,3）的平均
    }
