############################################################################################################
# 对比train2.py增添了以下功能：
# 1. 增加了对模型输出为元组的处理，使其能够支持adaptive_vit等返回多个输出的模型
############################################################################################################

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import os
from utils import AverageMeter, accuracy
from model import model_dict
import numpy as np
import time
import random
import wandb
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="deit")
parser.add_argument("--pre_trained", type=bool, default=True)
parser.add_argument("--classes_num", type=int, default=4)
parser.add_argument("--dataset", type=str, default="/kaggle/input/data-adni")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=33)
# parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--gpu-id", type=str, default="0, 1")
parser.add_argument("--print_freq", type=int, default=1)
parser.add_argument("--exp_postfix", type=str, default="seed33")
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4")
parser.add_argument("--project_name", type=str, default="ADNI-Classification")

args = parser.parse_args()


def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch(seed=args.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

exp_name = args.exp_postfix
exp_path = "./report/{}/{}/{}".format(args.dataset, args.model_names, exp_name)
os.makedirs(exp_path, exist_ok=True)

# 数据转换
transform_train = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize([256, 256]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.3738, 0.3738, 0.3738),
                         (0.3240, 0.3240, 0.3240))
])

transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.3738, 0.3738, 0.3738),
                         (0.3240, 0.3240, 0.3240))
])

# 加载完整数据集
full_dataset = datasets.ImageFolder(root=args.dataset, transform=transform_train)

# 计算划分大小
total_size = len(full_dataset)
train_size = int(0.7 * total_size)  # 70% 用于训练
val_size = int(0.15 * total_size)  # 15% 用于验证
test_size = total_size - train_size - val_size  # 剩余用于测试

# 随机划分数据集
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(args.seed)
)

# 为验证和测试集设置正确的transform
val_dataset.dataset.transform = transform_test
test_dataset.dataset.transform = transform_test

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


def train_one_epoch(model, optimizer, train_loader):
    model.train()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    for (inputs, targets) in tqdm(train_loader, desc="train"):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)
        # 检查模型输出是否为元组，如果是则取第一个元素（通常是主要的分类输出）
        out = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = F.cross_entropy(out, targets)
        loss_recorder.update(loss.item(), n=inputs.size(0))
        acc = accuracy(out, targets)[0]
        acc_recorder.update(acc.item(), n=inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = loss_recorder.avg
    acces = acc_recorder.avg

    return losses, acces


def evaluation(model, test_loader):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            outputs = model(img)
            # 检查模型输出是否为元组，如果是则取第一个元素
            out = outputs[0] if isinstance(outputs, tuple) else outputs

            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))
    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces


def train(model, optimizer, train_loader, val_loader, scheduler):
    since = time.time()
    best_acc = -1
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")

    # 在训练开始时检查模型输出结构
    with torch.no_grad():
        sample_input = next(iter(train_loader))[0][:1].cuda() if torch.cuda.is_available() else \
        next(iter(train_loader))[0][:1]
        outputs = model(sample_input)
        print("Model output type:", type(outputs))
        if isinstance(outputs, tuple):
            print("Output tuple length:", len(outputs))
            print("First output shape:", outputs[0].shape)

    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(
            model, optimizer, train_loader
        )
        val_losses, val_acces = evaluation(model, val_loader)

        if val_acces > best_acc:
            best_acc = val_acces
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=val_acces)
            name = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)

        scheduler.step()

        # 使用wandb记录训练指标
        wandb.log({
            "train_loss": train_losses,
            "train_accuracy": train_acces,
            "val_loss": val_losses,
            "val_accuracy": val_acces,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        })

        if (epoch + 1) % args.print_freq == 0:
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}\n".format(
                epoch + 1,
                args.model_names,
                train_losses,
                train_acces,
                val_losses,
                val_acces,
            )
            print(msg)
            f.write(msg)
            f.flush()

    # 在训练结束后评估测试集性能
    test_losses, test_acces = evaluation(model, test_loader)

    # 记录最终测试集性能到wandb
    wandb.log({
        "final_test_loss": test_losses,
        "final_test_accuracy": test_acces,
        "best_validation_accuracy": best_acc
    })

    msg_test = "Final test performance - loss:{:.2f} acc:{:.2f}\n".format(test_losses, test_acces)
    msg_best = "model:{} best validation acc:{:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "training time: {}".format(time.time() - since)
    print(msg_test)
    print(msg_best)
    f.write(msg_test)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == "__main__":
    # 初始化wandb
    wandb.init(
        project=args.project_name,
        name=f"{args.model_names}_{args.exp_postfix}",
        config={
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "model": args.model_names,
            "optimizer": "Adam",
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "dataset": args.dataset
        }
    )

    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)  # 添加并行封装
        model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

    train(model, optimizer, train_loader, val_loader, scheduler)

    # 结束wandb运行
    wandb.finish()