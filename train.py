import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import wandb
from evaluate_val import evaluate_val
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path(r'D:\projects\2026\graduation_project\U-Net模型训练与构建\Pytorch-UNet-master\Pytorch-UNet-master\data\imgs')
dir_mask = Path(r'D:\projects\2026\graduation_project\U-Net模型训练与构建\Pytorch-UNet-master\Pytorch-UNet-master\data\masks')
dir_checkpoint = Path(r'D:\projects\2026\graduation_project\U-Net模型训练与构建\Pytorch-UNet-master\Pytorch-UNet-master\checkpoints')

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, epochs_list):
    """绘制训练损失和验证精度曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制损失曲线
    ax1.plot(epochs_list, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o')
    ax1.plot(epochs_list, val_losses, 'orange', linewidth=2, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制精度曲线
    ax2.plot(epochs_list, train_accuracies, 'b-', linewidth=2, label='Training Accuracy', marker='o')
    ax2.plot(epochs_list, val_accuracies, 'orange', linewidth=2, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = './training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f'Training history plot saved to {output_path}')
    plt.close()


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        val_percent: float = 0.3,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset，定义dataset变量
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    try:
        # 创建一个不带增强的全量 dataset 用于扫描 unique values 等初始化
        temp_dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except:
        temp_dataset = BasicDataset(dir_img, dir_mask, img_scale)

    all_ids = temp_dataset.ids  # 获取所有ID
    n_val = int(len(all_ids) * val_percent)
    n_train = len(all_ids) - n_val
    train_ids, val_ids = random_split(all_ids, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 【核心】分别实例化训练和验证 Dataset
    # 仅给训练集设置 augment=True
    train_set = BasicDataset(dir_img, dir_mask, img_scale, augment=True)
    train_set.ids = train_ids  # 手动覆盖 Subset 的 ID 逻辑

    val_set = BasicDataset(dir_img, dir_mask, img_scale, augment=False)
    val_set.ids = val_ids

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # 显式定义 validation Dice 为一个度量指标，并以 step 作为 x 轴
    experiment.define_metric("validation Dice", step_metric="step", summary="max")
    experiment.define_metric("train loss", step_metric="step", summary="min")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=1e-4,  # 适当的权重衰减防止过拟合
                            amsgrad=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # --- 修改后的 scheduler 定义 ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控指标（Dice）越大越好
        patience=15,  # 【修改这里】耐心值，建议从 3 改为 5 或 7
        factor=0.3,  # 【新增这里】学习率衰减倍数，从默认 0.1 改为 0.5（下降更温和）
        threshold=1e-4,  # 【新增这里】提升阈值，只有超过这个增量才算进步
        # verbose=True  # 【建议新增】这样你可以在控制台看到学习率下降的提示
    )  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 添加这些列表来追踪每个epoch的指标
    train_losses = []      # 每个epoch的平均训练损失
    val_losses = []        # 每个epoch的验证损失
    train_accuracies = []  # 每个epoch的训练精度
    val_accuracies = []    # 每个epoch的验证精度
    epochs_list = []       # epoch编号

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        num_train_batches = len(train_loader)
        previous_step = global_step
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.amp.autocast('cuda', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({"train loss": loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # 找到 Evaluation round 里的 experiment.log 部分，修改为：
                        # --- 在 train_model 函数中找到 Evaluation round 部分 ---

                        # 1. 运行验证函数获取分数
                        val_score = evaluate_val(model, val_loader, device, amp)
                         # 将张量转换为 Python 浮点数
                        if isinstance(val_score, torch.Tensor):
                            val_score = val_score.cpu().item()
                        
                        scheduler.step(val_score)
                        logging.info(f'Validation Dice score: {val_score}')

                        # 2. 【核心修改】只上传纯数值指标，彻底移除 wandb.Image
                        # 只要不传图像，就不会触发 [12, 128, 128] 的报错
                        experiment.log({
                            "learning rate": optimizer.param_groups[0]['lr'],
                            "validation Dice": val_score,  # 这是你想要的纵轴变量
                            "step": global_step,  # 这是你想要的横轴变量
                            "epoch": epoch
                        })

                        # 如果你还是想看直方图，可以保留这一句（直方图通常不会因为通道数报错）
                        if histograms:
                            experiment.log({**histograms, "step": global_step})

        # 在每个epoch结束时，计算并记录平均训练loss和验证指标
        avg_epoch_loss = epoch_loss / num_train_batches if num_train_batches > 0 else 0
        train_losses.append(avg_epoch_loss)

        # 计算该epoch的验证损失和精度
        val_loss = 0
        val_accuracy = 0
        num_val_batches = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        batch_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        batch_loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        batch_loss = criterion(masks_pred, true_masks)
                        batch_loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                val_loss += batch_loss.item()
                num_train_batches += 1
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        
        # 3. 计算验证精度（Dice分数）
        val_accuracy = evaluate_val(model, val_loader, device, amp)
        val_accuracy = val_accuracy.cpu().item() if isinstance(val_accuracy, torch.Tensor) else val_accuracy
        val_accuracies.append(val_accuracy)

        # 4. 计算训练精度（Dice分数）
        train_accuracy = evaluate_val(model, train_loader, device, amp)
        train_accuracy = train_accuracy.cpu().item() if isinstance(train_accuracy, torch.Tensor) else train_accuracy
        train_accuracies.append(train_accuracy)

        # 5. 记录 epoch 编号
        epochs_list.append(epoch)

        logging.info(f'Epoch {epoch} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                     f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    # 训练完成后绘制图表
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, epochs_list)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=5, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
