# Lab1-Pneumonia-Classification
The homework 1 of the lecture Research Project in Application of Artificial Intelligence on Medical Imaging. From student 314351016.


## Introduction

本專案利用深度學習技術進行胸腔 X 光影像的肺炎二元分類。專案利用 PyTorch 框架進行開發，建立自定義的 DataLoader，並使用 Data Augmentation 與 Weighted Cross-Entropy Loss 以克服類別不平衡問題。
實驗比較了 ResNet18 與 ResNet50 兩種網路架構，透過載入 ImageNet 預訓練權重進行微調，並重新初始化分類層以符合二元分類需求。


## Requirements

* Python 3.8+
* torch
* torchvision
* numpy
* matplotlib
* seaborn
* tqdm


## Dataset Preparation

本專案使用 Kaggle 平台提供的 Chest X-Ray Images 資料集。
請下載並解壓縮資料集，確保資料夾結構如下：

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/


## How to Run

專案包含兩個主要的 Python Scripts：train.py 與 inference.py。

*Training*

train.py 會自動載入資料集，訓練 ResNet18 與 ResNet50。過程中會儲存具備最高測試準確度的模型權重，並在訓練結束後輸出 Accuracy 曲線、F1-score 曲線以及混淆矩陣圖表。

python train.py --dataset "C:\Users\lueri\Desktop\chest_xray"

自定義參數指令：
python train.py --dataset "資料路徑" --num_epochs 30 --batch_size 32 --lr 0.00005 --wd 0.0001

參數說明：
--dataset: 資料集根目錄的絕對路徑
--num_epochs: 訓練總回合數 (預設: 30)
--batch_size: 批次大小 (預設: 32)
--lr: 初始學習率 (預設: 5e-5)
--wd: 權重衰減 / L2 正規化 (預設: 1e-4)
--resize: 影像重塑尺寸 (預設: 224)
--degree: 資料擴增之隨機旋轉角度上限 (預設: 15)

*Inference*

執行 inference.py 可讀取已訓練好的權重檔案，針對測試集進行獨立評估，並計算 Accuracy、Precision、Recall 與 F1-score。

ResNet18 模型：
python inference.py --model_name resnet18 --weights best_resnet18_weights.pt --dataset "..\chest_xray"

ResNet50 模型：
python inference.py --model_name resnet50 --weights best_resnet50_weights.pt --dataset "..\chest_xray"

可調用參數：
--model_name: 指定要載入的模型架構，僅接受 resnet18 或 resnet50。
--weights: 指定已儲存之模型權重檔案 (.pt) 的路徑。
--dataset: 測試資料集根目錄的路徑。


## Outputs

成功執行後，專案目錄下將會生成以下檔案：
權重檔：best_resnet18_weights.pt, best_resnet50_weights.pt
學習曲線圖：[model]_accuracy_curve.png, [model]_f1_score_curve.png
混淆矩陣：[model]_confusion_matrix.png, [model]_inference_confusion_matrix.png
