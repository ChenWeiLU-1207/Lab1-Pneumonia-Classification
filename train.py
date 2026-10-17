import os
import gc
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

# ================= 關鍵修改開始 =================
def plot_training_curves(train_acc_list, train_f1_list, model_name):
    """繪製 Training Accuracy & F1-score 曲線 (對應新版 Spec 3.b.i)"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Accuracy (%)', color=color)
    ax1.plot(train_acc_list, label='Train Accuracy', color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Training F1-Score', color=color)  
    ax2.plot(train_f1_list, label='Train F1-Score', color=color, marker='^', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{model_name.upper()} Training Curves')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'{model_name}_training_curves.png')
    plt.close()

def plot_validation_curves(val_acc_list, val_f1_list, model_name):
    """繪製 Validation Accuracy & F1-score 曲線 (對應新版 Spec 3.b.ii Bonus)"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:orange'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Accuracy (%)', color=color)
    ax1.plot(val_acc_list, label='Val Accuracy', color=color, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Validation F1-Score', color=color)  
    ax2.plot(val_f1_list, label='Val F1-Score', color=color, marker='D', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{model_name.upper()} Validation Curves (Bonus)')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'{model_name}_validation_curves.png')
    plt.close()
# ================= 關鍵修改結束 =================

def plot_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='rocket', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'{model_name.upper()} Final Test Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def train(device, train_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, model_name):
    best_acc = 0.0
    best_c_matrix = []
    
    # 儲存紀錄的 List 更新
    train_acc_list, train_f1_list = [], []
    test_acc_list, test_f1_list = [], []

    for epoch in range(1, num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0    
            
            for data in tqdm(train_loader, desc=f"[{model_name.upper()}] Epoch {epoch}/{num_epochs} Training"):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            
            # 計算 Training 的 F1-Score
            t_precision = tp / (tp+fp) if (tp+fp) > 0 else 0
            t_recall = tp / (tp+fn) if (tp+fn) > 0 else 0
            train_f1 = (2*t_precision*t_recall) / (t_precision+t_recall) if (t_precision+t_recall) > 0 else 0
            
            print(f'\nEpoch: {epoch}')
            print(f'↳ Loss: {avg_loss:.4f}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}% | Training F1: {train_f1:.4f}')
            print(f'↳ Current LR: {optimizer.param_groups[0]["lr"]:.6f}')

        test_acc, test_f1, c_matrix = test(device, test_loader, model)
        scheduler.step()

        # 紀錄歷史軌跡
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_c_matrix = c_matrix
            torch.save(model.state_dict(), f'best_{model_name}_weights.pt')
            print(f'* Updated New Weight (Highest Test Acc: {best_acc:.2f}%)')

    return train_acc_list, train_f1_list, test_acc_list, test_f1_list, best_c_matrix

def test(device, test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tn), int(fp)],
                    [int(fn), int(tp)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn) if (tp+fn) > 0 else 0
        precision = tp / (tp+fp) if (tp+fp) > 0 else 0
        f1_score = (2*tp) / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
        
        print (f'↳ Val Recall: {recall:.4f}, Val Precision: {precision:.4f}, Val F1-score: {f1_score:.4f}')
        print (f'↳ Val Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=32) 
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=1e-4) 
    parser.add_argument('--dataset', type=str, required=False, default=r"C:\Users\lueri\Desktop\chest_xray")
    parser.add_argument('--degree', type=int, default=15)
    parser.add_argument('--resize', type=int, default=224)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.RandomRotation(args.degree),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_architectures = ['resnet18', 'resnet50']

    for model_name in model_architectures:
        print(f"\n{'='*50}")
        print(f"Start Model Training: {model_name.upper()}")
        print(f"{'='*50}")

        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, args.num_classes)
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346])).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        train_acc, train_f1, val_acc, val_f1, best_c_matrix = train(
            device, train_loader, test_loader, model, criterion, optimizer, scheduler, args.num_epochs, model_name
        )

        # 呼叫更新後的繪圖函數
        plot_training_curves(train_acc, train_f1, model_name)
        plot_validation_curves(val_acc, val_f1, model_name)
        plot_confusion_matrix(best_c_matrix, model_name)

        print(f"{model_name.upper()} Finish Training. (Images saved to local directory)")

        del model, optimizer, criterion, scheduler
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()