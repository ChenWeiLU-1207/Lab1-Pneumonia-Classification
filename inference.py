import os
import warnings
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

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

def plot_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='rocket', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'{model_name.upper()} Inference Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_inference_confusion_matrix.png')
    plt.close()

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
        
        print(f'↳ Inference Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print(f'↳ Inference Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet18', 'resnet50'], help="Choose the model architecture")
    parser.add_argument('--weights', type=str, required=True, help="Path to the saved .pt weights file")
    parser.add_argument('--dataset', type=str, required=False, default=r"C:\Users\lueri\Desktop\chest_xray")
    parser.add_argument('--batch_size', type=int, required=False, default=32) 
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} for Inference ##')
    print(f'## Loading Model: {args.model_name.upper()} | Weights: {args.weights} ##')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_name == 'resnet18':
        model = models.resnet18()
    elif args.model_name == 'resnet50':
        model = models.resnet50()

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, args.num_classes)
    )
    
    # 載入指定的權重檔案
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model = model.to(device)

    _, _, c_matrix = test(device, test_loader, model)
    plot_confusion_matrix(c_matrix, args.model_name)
    print(f"Inference complete. Confusion matrix saved as {args.model_name}_inference_confusion_matrix.png")