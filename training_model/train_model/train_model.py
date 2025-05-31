import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
import joblib
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier
import xgboost as xgb
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from scipy.stats import mode
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 檢查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        # 計算3層網絡的層大小 (all-mpnet-base-v2 通常是 768 維)
        # 假設 input_dim 是 768
        layer1_size = int(input_dim * 1.5)  # 例如 768 * 1.5 = 1152
        layer2_size = int(layer1_size * 0.66) # 例如 1152 * 0.66 = 760
        layer3_size = int(layer2_size * 0.5)  # 例如 760 * 0.5 = 380
        
        # 定義網絡層 (3層)
        self.network = nn.Sequential(
            # 第一層
            nn.Linear(input_dim, layer1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(layer1_size),
            
            # 第二層
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(layer2_size),
            
            # 第三層
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(layer3_size),
            
            # 輸出層
            nn.Linear(layer3_size, 2)
        )
        
        # 保存層大小供打印使用
        self.layers_info = {
            'input': input_dim,
            'layer1': layer1_size,
            'layer2': layer2_size,
            'layer3': layer3_size,
            'output': 2
        }
        
    def forward(self, x):
        return self.network(x)

def load_data():
    print("載入標記資料...")
    # 直接從 npy 文件加載標籤
    print("載入標籤...")
    labels = np.load('data/proc/labels.npy')
    
    print("反轉正負類別...")
    labels = 1 - labels  # 反轉標籤：0變1，1變0
    
    print("載入句子向量...")
    # 讀取句子向量
    embeddings = np.load('data/proc/embeddings.npy')
    
    print(f"數據集大小: {len(labels)} 個樣本")
    print(f"特徵維度: {embeddings.shape[1]}")
    
    return labels, embeddings

def batch_generator(X, y, batch_size=10000):
    """生成批次數據的生成器"""
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def find_best_threshold(y_true, y_prob):
    """找到最佳的分類閾值"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def save_checkpoint(state, fold_idx, epoch, model_dir):
    """保存檢查點"""
    checkpoint_dir = Path(model_dir) / 'checkpoints' / f'fold_{fold_idx}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'epoch_{epoch}_checkpoint.pt'
    torch.save(state, checkpoint_path)
    print(f"保存檢查點到: {checkpoint_path}")

def train_fold(X_train, y_train, X_val, y_val, fold_idx, input_dim, pos_weight, n_splits):
    """訓練單個折疊的所有模型"""
    # 創建模型目錄
    model_dir = Path('data/models')
    model_dir.mkdir(exist_ok=True)
    
    # 準備數據加載器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # 轉換驗證集為張量
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    # 初始化模型
    mlp_model = MLPModel(input_dim).to(device)
    
    # 打印模型架構信息
    print("\n模型架構：")
    for name, size in mlp_model.layers_info.items():
        print(f"{name}: {size} 神經元")
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, pos_weight]).to(device))
    optimizer = optim.AdamW(mlp_model.parameters(), lr=0.001, weight_decay=0.01)
    # scheduler = optim.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    scaler = GradScaler()
    
    # 訓練 MLP
    best_f1 = 0
    epochs = 300
    early_stopping_patience = 10
    no_improve_count = 0
    
    print(f"\n訓練折疊 {fold_idx + 1}/{n_splits} MLP...")
    for epoch in range(epochs):
        mlp_model.train()
        total_loss = 0
        
        # 使用tqdm顯示進度條
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                
                # 使用混合精度訓練
                with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    outputs = mlp_model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 驗證
        mlp_model.eval()
        with torch.no_grad(), autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            outputs = mlp_model(X_val_tensor)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            current_f1 = f1_score(y_val, predictions)
            current_accuracy = accuracy_score(y_val, predictions)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val F1: {current_f1:.4f}, Val Acc: {current_accuracy:.4f}")
        
        # 保存檢查點
        if (epoch + 1) % 10 == 0:  # 每10輪保存一次
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mlp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss/len(train_loader),
                'f1': current_f1,
                'accuracy': current_accuracy,
                'best_f1': best_f1
            }
            save_checkpoint(checkpoint, fold_idx, epoch + 1, model_dir)
        
        scheduler.step(current_f1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_mlp_state = mlp_model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= early_stopping_patience:
            print("Early stopping triggered")
            break
    
    # 載入最佳 MLP 模型
    mlp_model.load_state_dict(best_mlp_state)
    
    # 訓練 CatBoost
    print("\n訓練 CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=300,  # 增加迭代次數
        learning_rate=0.01,
        depth=8,  # 增加深度
        class_weights={0: 1, 1: pos_weight},
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type='CPU'
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    # 訓練 XGBoost
    print("\n訓練 XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=8,
        scale_pos_weight=pos_weight,
        random_state=42,
        tree_method='hist',
        enable_categorical=False,
        use_label_encoder=False,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=50
    )
    
    # 使用 early_stopping 進行訓練
    eval_set = [(X_val, y_val)]
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # 獲取所有模型的預測概率
    mlp_model.eval()
    with torch.no_grad(), autocast(device_type=device.type, enabled=torch.cuda.is_available()):
        mlp_probs = torch.softmax(mlp_model(X_val_tensor), dim=1)[:, 1].cpu().numpy()
    
    cat_probs = cat_model.predict_proba(X_val)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    
    # 平均預測概率
    ensemble_probs = (mlp_probs + cat_probs + xgb_probs) / 3
    
    return {
        'mlp': mlp_model,
        'cat': cat_model,
        'xgb': xgb_model,
        'probs': ensemble_probs,
        'mlp_state': best_mlp_state
    }

def train_model(model_type='ensemble'):
    print("載入資料...")
    labels, embeddings = load_data()
    
    # 釋放一些記憶體
    gc.collect()
    
    # 初始化 StandardScaler
    print("正在標準化特徵...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # 設置交叉驗證
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 計算正樣本權重
    pos_weight = np.sum(labels == 0) / np.sum(labels == 1)
    print(f"Positive class weight (for new positive class): {pos_weight}")
    
    # 儲存每個折疊的模型和預測
    fold_models = []
    all_val_probs = np.zeros(len(labels))
    all_val_indices = np.zeros(len(labels))
    
    # 對每個折疊進行訓練
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings_scaled, labels)):
        print(f"\n開始訓練第 {fold_idx + 1}/{n_splits} 折...")
        
        X_train, X_val = embeddings_scaled[train_idx], embeddings_scaled[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # 訓練模型
        fold_result = train_fold(X_train, y_train, X_val, y_val, fold_idx, embeddings_scaled.shape[1], pos_weight, n_splits)
        
        # 儲存預測和模型
        all_val_probs[val_idx] = fold_result['probs']
        all_val_indices[val_idx] = 1
        fold_models.append(fold_result)
        
        print(f"完成第 {fold_idx + 1} 折訓練")
    
    # 找到最佳閾值
    best_threshold, best_f1 = find_best_threshold(labels, all_val_probs)
    print(f"\n最佳閾值: {best_threshold:.3f}, 最佳 F1: {best_f1:.3f}")
    
    # 使用最佳閾值進行預測
    final_predictions = (all_val_probs >= best_threshold).astype(int)
    
    # 計算最終指標
    precision = precision_score(labels, final_predictions)
    recall = recall_score(labels, final_predictions)
    f1 = f1_score(labels, final_predictions)
    accuracy = accuracy_score(labels, final_predictions)
    
    print("\n最終模型效能：")
    print(f"Precision (for new positive class): {precision:.3f}")
    print(f"Recall (for new positive class): {recall:.3f}")
    print(f"F1 Score (for new positive class): {f1:.3f}")
    print(f"Accuracy (for new positive class): {accuracy:.3f}")
    
    # 無論 F1 分數如何都儲存模型
    print("\n正在儲存模型...")
    model_dir = Path('data/models')
    model_dir.mkdir(exist_ok=True)
    
    # 使用時間戳作為模型版本
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f'ensemble_highlighter_v1_{timestamp}.pkl'
    
    model_data = {
        'fold_models': fold_models,
        'scaler': scaler,
        'best_threshold': best_threshold,
        'input_dim': embeddings_scaled.shape[1],
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        },
        'training_info': {
            'timestamp': timestamp,
            'n_splits': n_splits,
            'pos_weight': pos_weight
        }
    }
    
    torch.save(model_data, model_path)
    print(f"模型已儲存到：{model_path}")
    
    if f1 >= 0.55:  # 降低目標 F1 分數到 0.55 (此F1針對新的正類)
        print("\n模型達到目標 F1 分數 (≥ 0.55) (針對新的正類)")
    else:
        print("\n警告：模型未達到目標 F1 分數 (0.55) (針對新的正類)，建議調整參數後重試")

if __name__ == '__main__':
    train_model('ensemble') 