import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from PIL import Image
import os
import argparse

# 初始化分布式环境
def init_distributed_mode(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

# 创建 MsDataset 的 PyTorch 数据集适配器
class MsDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, ms_dataset, transform=None):
        self.ms_dataset = ms_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ms_dataset)

    def __getitem__(self, idx):
        sample = self.ms_dataset[idx]
        image = Image.open(sample['image:FILE']).convert('RGB')
        label = sample['category']

        if self.transform:
            image = self.transform(image)

        return image, label

# 选择模型
def get_model(model_name):
    if model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        return models.resnet101(pretrained=True)
    elif model_name == 'vit_ti':
        from torchvision.models import vit_b_16
        return vit_b_16(pretrained=True)
    elif model_name == 'vit_large':
        from torchvision.models import vit_l_16
        return vit_l_16(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# 训练主函数
def main():
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--epochs", type=int, default=10, help="number of total epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--local-rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    args = parser.parse_args()
    init_distributed_mode(args)
    device = torch.device("cuda", torch.distributed.get_rank())

    transform = transforms.Compose([
        # 继续其他预处理
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载数据集（需确保已登录 ModelScope 并下载数据集）
    try:
        ms_train_dataset = MsDataset.load('mini_imagenet100', namespace='tany0699', subset_name='default', split='train')
        ms_val_dataset = MsDataset.load('mini_imagenet100', namespace='tany0699', subset_name='default', split='validation')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    train_dataset = MsDatasetAdapter(ms_train_dataset, transform=transform)
    val_dataset = MsDatasetAdapter(ms_val_dataset, transform=transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    model_names = ['vit_ti', 'vit_large']

    for model_name in model_names:
        model = get_model(model_name)
        model = model.to(device)
        model = DDP(model)  # 自动分配所有 GPU

        # 动态调整学习率
        if model_name in ['vit_ti', 'vit_large']:
            lr = 1e-5
        else:
            lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        log_dir = f'./runs/{model_name}'
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 0:
                    avg_loss = running_loss / (i + 1)
                    avg_acc = 100 * correct / total
                    global_step = epoch * len(train_loader) + i
                    writer.add_scalar(f'Loss/{model_name}', avg_loss, global_step)
                    writer.add_scalar(f'Accuracy/{model_name}', avg_acc, global_step)

                    if i == 0:
                        img_grid = make_grid(inputs[:16])
                        writer.add_image('Training Images', img_grid, global_step)

            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

            if epoch % 5 == 0:
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * correct_val / total_val:.2f}%")
        writer.close()

if __name__ == '__main__':
    main()
