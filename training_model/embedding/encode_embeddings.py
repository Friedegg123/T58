# scripts/encode_embeddings.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import torch
import os
import shutil

def get_device():
    """選擇最佳可用的設備"""
    if torch.cuda.is_available():
        print("使用 CUDA GPU")
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("使用 Apple M1/M2 GPU")
        return torch.device('mps')
    else:
        print("使用 CPU")
        return torch.device('cpu')

def load_sentences():
    """從標記文件中加載所有句子和標籤"""
    print("正在讀取標記數據...")
    sentences = []
    labels = []
    
    input_file = Path("drive/MyDrive/AI/labeled_cnn.jsonl")
    if not input_file.exists():
        raise FileNotFoundError("找不到標記文件 data/proc/labeled.jsonl")
    
    # 首先計算總行數以顯示進度
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="讀取句子"):
            obj = json.loads(line)
            sentences.append(obj["s"])
            labels.append(obj["y"])
    
    print(f"總共讀取了 {len(sentences)} 個句子")
    return sentences, np.array(labels)

def get_last_processed_batch():
    """獲取最後處理的批次編號"""
    embed_dir = Path("data/proc/embeddings")
    if not embed_dir.exists():
        return -1
    
    batch_files = list(embed_dir.glob("batch_*.npy"))
    if not batch_files:
        return -1
    
    batch_nums = [int(f.stem.split('_')[1]) for f in batch_files]
    return max(batch_nums) if batch_nums else -1

def process_batch(model, sentences, labels, batch_idx, batch_size, device):
    """處理並保存一個批次的數據"""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(sentences))
    
    # 獲取當前批次的句子
    batch_sentences = sentences[start_idx:end_idx]
    batch_labels = labels[start_idx:end_idx]
    
    # 生成嵌入向量
    batch_embeddings = model.encode(
        batch_sentences,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    
    # 保存批次數據
    embed_dir = Path("data/proc/embeddings")
    embed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(embed_dir / f"batch_{batch_idx}.npy", batch_embeddings)
    np.save(embed_dir / f"labels_{batch_idx}.npy", batch_labels)
    
    # 保存批次信息
    with open(embed_dir / "batch_info.txt", "a") as f:
        f.write(f"Batch {batch_idx}: {start_idx} to {end_idx} ({len(batch_sentences)} sentences)\n")
    
    return end_idx - start_idx

def combine_results():
    """合併所有批次的結果"""
    print("\n正在合併所有批次的結果...")
    embed_dir = Path("data/proc/embeddings")
    
    # 獲取所有批次文件並排序
    batch_files = sorted(embed_dir.glob("batch_*.npy"), 
                        key=lambda x: int(x.stem.split('_')[1]))
    label_files = sorted(embed_dir.glob("labels_*.npy"),
                        key=lambda x: int(x.stem.split('_')[1]))
    
    if not batch_files:
        raise FileNotFoundError("沒有找到批次文件")
    
    # 讀取第一個批次以獲取維度信息
    first_batch = np.load(batch_files[0])
    total_embeddings = []
    total_labels = []
    
    # 合併所有批次
    for batch_file, label_file in tqdm(zip(batch_files, label_files), 
                                     desc="合併批次", 
                                     total=len(batch_files)):
        embeddings = np.load(batch_file)
        labels = np.load(label_file)
        total_embeddings.append(embeddings)
        total_labels.append(labels)
    
    # 將所有批次連接成一個數組
    final_embeddings = np.concatenate(total_embeddings)
    final_labels = np.concatenate(total_labels)
    
    # 保存最終結果
    np.save("data/proc/embeddings.npy", final_embeddings)
    np.save("data/proc/labels.npy", final_labels)
    
    # 輸出統計信息
    print(f"\n最終結果：")
    print(f"嵌入向量數量: {final_embeddings.shape[0]}")
    print(f"向量維度: {final_embeddings.shape[1]}")
    print(f"標籤數量: {len(final_labels)}")
    
    # 計算正例比例
    positive_labels = np.sum(final_labels == 1)
    print(f"\n數據集統計:")
    print(f"正例（標籤為1）: {positive_labels} ({positive_labels/len(final_labels)*100:.2f}%)")
    print(f"負例（標籤為0）: {len(final_labels)-positive_labels} ({(1-positive_labels/len(final_labels))*100:.2f}%)")
    
    # 詢問是否刪除臨時文件
    return final_embeddings, final_labels

def cleanup_temp_files():
    """清理臨時文件"""
    print("\n清理臨時文件...")
    embed_dir = Path("data/proc/embeddings")
    if embed_dir.exists():
        shutil.rmtree(embed_dir)
    print("臨時文件已清理完成")

def main():
    # 確保輸出目錄存在
    Path("data/proc").mkdir(parents=True, exist_ok=True)
    
    # 1. 讀取所有句子和標籤
    sentences, labels = load_sentences()
    
    # 2. 檢查上次處理的批次
    last_batch = get_last_processed_batch()
    
    # 3. 載入預訓練模型
    print("\n正在加載模型...")
    device = get_device()
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    # 4. 設置批次大小和計算總批次數
    batch_size = 10000  # 每個批次處理 10000 個句子
    total_batches = (len(sentences) + batch_size - 1) // batch_size
    
    # 5. 處理每個批次
    print(f"\n開始處理 {total_batches} 個批次...")
    
    try:
        for batch_idx in range(last_batch + 1, total_batches):
            print(f"\n處理批次 {batch_idx + 1}/{total_batches}")
            processed = process_batch(model, sentences, labels, batch_idx, batch_size, device)
            print(f"完成處理 {processed} 個句子")
        
        # 6. 合併所有批次的結果
        final_embeddings, final_labels = combine_results()
        
        # 7. 清理臨時文件
        cleanup_temp_files()
        
        print("\n所有處理完成！")
        
    except KeyboardInterrupt:
        print("\n\n檢測到中斷！")
        print("已保存的批次仍然可用，重新運行腳本將從上次中斷的地方繼續。")
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
