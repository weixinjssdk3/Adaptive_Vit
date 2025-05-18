############################################################################################################
# 对比train5.py增添了以下功能：
# 1. 增加了精确率(Precision)、召回率(Recall)和F1分数评价指标
# 2. 增加了计算量(GFLOPs)评价指标
# 3. 增加了推理时间(ms)评价指标
# 4. 增加了ROC曲线和AUC值评价指标
# 5. 增加了混淆矩阵分析
############################################################################################################

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from thop import profile  # 用于计算GFLOPs

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="standard_vit")
parser.add_argument("--pre_trained", type=bool, default=True)
parser.add_argument("--classes_num", type=int, default=4)
parser.add_argument("--dataset", type=str, default="/kaggle/input/data-tumor")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=33)
parser.add_argument("--gpu-id", type=str, default="0, 1")
parser.add_argument("--print_freq", type=int, default=1)
parser.add_argument("--exp_postfix", type=str, default="seed33")
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4")
parser.add_argument("--project_name", type=str, default="Renal-Classification")

args = parser.parse_args()

def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=args.seed)
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

# ===================== 使用ImageFolder加载数据 =====================
train_dir = os.path.join(args.dataset, 'train')
val_dir = os.path.join(args.dataset, 'valid')
test_dir = os.path.join(args.dataset, 'test')

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

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

def evaluation(model, test_loader, compute_metrics=False):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    all_preds = []
    all_targets = []
    all_probs = []
    inference_times = []

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            if compute_metrics:
                start_time = time.time()
            outputs = model(img)
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            if compute_metrics:
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time / img.size(0))
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))
            if compute_metrics:
                _, preds = torch.max(out, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(label.cpu().numpy())
                all_probs.append(F.softmax(out, dim=1).cpu().numpy())
    losses = loss_recorder.avg
    acces = acc_recorder.avg
    metrics = {}
    if compute_metrics:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.concatenate(all_probs, axis=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        avg_inference_time = np.mean(inference_times)
        cm = confusion_matrix(all_targets, all_preds)
        n_classes = args.classes_num
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((all_targets == i).astype(int), all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_roc_auc = auc(all_fpr, mean_tpr)
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_inference_time': avg_inference_time,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'macro_roc_auc': macro_roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'all_fpr': all_fpr,
            'mean_tpr': mean_tpr
        }
    return losses, acces, metrics

def calculate_gflops(model):
    input_tensor = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model = model.cuda()
    macs, params = profile(model, inputs=(input_tensor,))
    gflops = macs / 1e9
    return gflops, params

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(exp_path, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    return confusion_matrix_path

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, roc_auc, n_classes):
    plt.figure(figsize=(10, 8))
    
    # 绘制每个类别的ROC曲线
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    # 绘制宏平均ROC曲线
    plt.plot(fpr['macro'], tpr['macro'], 'k--',
             label='Macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['macro']), lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 保存图像并上传到wandb
    roc_curve_path = os.path.join(exp_path, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()
    
    return roc_curve_path


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
    
    # 计算模型的GFLOPs
    gflops, params = calculate_gflops(model)
    print(f"Model GFLOPs: {gflops:.4f}, Parameters: {params/1e6:.2f}M")
    
    # 记录GFLOPs到wandb
    wandb.log({
        "GFLOPs": gflops,
        "Parameters(M)": params/1e6
    })

    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(
            model, optimizer, train_loader
        )
        val_losses, val_acces, _ = evaluation(model, val_loader)

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

    # 在训练结束后评估测试集性能，并计算额外的评价指标
    test_losses, test_acces, metrics = evaluation(model, test_loader, compute_metrics=True)
    
    # 获取类别名称
    class_names = [str(i) for i in range(args.classes_num)]
    
    # 绘制混淆矩阵
    cm_path = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # 准备ROC曲线数据
    fpr_dict = metrics['fpr']
    tpr_dict = metrics['tpr']
    roc_auc_dict = metrics['roc_auc']
    
    # 添加宏平均ROC曲线数据
    fpr_dict['macro'] = metrics['all_fpr']
    tpr_dict['macro'] = metrics['mean_tpr']
    roc_auc_dict['macro'] = metrics['macro_roc_auc']
    
    # 绘制ROC曲线
    roc_path = plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, args.classes_num)
    
    # 记录最终测试集性能到wandb
    wandb.log({
        "final_test_loss": test_losses,
        "final_test_accuracy": test_acces,
        "best_validation_accuracy": best_acc,
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1_score": metrics['f1'],
        "avg_inference_time_ms": metrics['avg_inference_time'],
        "macro_auc": metrics['macro_roc_auc'],
        "confusion_matrix": wandb.Image(cm_path),
        "roc_curve": wandb.Image(roc_path)
    })

    msg_test = "Final test performance - loss:{:.2f} acc:{:.2f}\n".format(test_losses, test_acces)
    msg_metrics = "Precision:{:.4f} Recall:{:.4f} F1:{:.4f} Inference time:{:.2f}ms AUC:{:.4f}\n".format(
        metrics['precision'], metrics['recall'], metrics['f1'], 
        metrics['avg_inference_time'], metrics['macro_roc_auc']
    )
    msg_best = "model:{} best validation acc:{:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "training time: {}\n".format(time.time() - since)
    print(msg_test)
    print(msg_metrics)
    print(msg_best)
    f.write(msg_test)
    f.write(msg_metrics)
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