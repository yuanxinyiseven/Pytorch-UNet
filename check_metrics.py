import torch
from unet import UNet
from evaluate import evaluate
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader


def main():
    # 1. 加载训练好的 checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=4)

    checkpoint = torch.load('./checkpoints/checkpoint_epoch11.pth', map_location=device)

    # 移除多余的键
    if 'mask_values' in checkpoint:
        del checkpoint['mask_values']

    net.load_state_dict(checkpoint)
    net.to(device)

    # 2. 加载验证集 (建议设置 num_workers=0 来彻底规避 Windows 下的进程问题)
    val_set = BasicDataset('./data/test/imgs', './data/test/masks', scale=0.5)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    # 3. 运行评估
    results = evaluate(net, val_loader, device, amp=True)

    # 4. 打印格式化结果
    class_names = ['Unchanged', 'Logging', 'Fire', 'Non-Forest']
    print(f"\nOverall Accuracy (OA): {results['OA']:.4f}")
    print("-" * 60)
    print(f"{'Class':<15} | {'PA':<8} | {'UA':<8} | {'F1/Dice':<8} | {'IoU':<8}")
    for i, name in enumerate(class_names):
        print(
            f"{name:<15} | {results['PA'][i]:.4f} | {results['UA'][i]:.4f} | {results['F1'][i]:.4f} | {results['IoU'][i]:.4f}")


if __name__ == '__main__':
    main()