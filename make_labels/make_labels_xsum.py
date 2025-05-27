from pathlib import Path
from datasets import load_dataset
from nltk import sent_tokenize, download
from sentence_transformers import SentenceTransformer
import json, tqdm
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from tqdm.auto import tqdm
import time
import random

# 設置環境變數以避免 CUDA/MPS 相關問題
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_model_with_retry(task_id: int, max_retries: int = 5) -> SentenceTransformer:
    """嘗試加載模型，如果失敗則重試"""
    for attempt in range(max_retries):
        try:
            print(f"\n任務 {task_id} 正在加載模型... (嘗試 {attempt + 1}/{max_retries})")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            device = torch.device('cpu')
            model = model.to(device)
            print(f"任務 {task_id} 模型加載完成！")
            return model, device
        except Exception as e:
            if attempt < max_retries - 1:
                delay = random.uniform(5, 15)  # 隨機延遲5-15秒
                print(f"任務 {task_id} 加載失敗: {str(e)}")
                print(f"等待 {delay:.1f} 秒後重試...")
                time.sleep(delay)
            else:
                raise Exception(f"任務 {task_id} 在 {max_retries} 次嘗試後仍然失敗")
    return None, None

def process_task(task_data: List[Dict], task_id: int) -> None:
    """處理一個任務批次並將結果寫入臨時文件"""
    task_progress = None
    
    try:
        # 下載 NLTK 數據（每個線程獨立）
        download("punkt", quiet=True)
        
        # 加載模型（帶重試機制）
        model, device = load_model_with_retry(task_id)
        
        # 為每個任務創建獨立的進度條
        task_progress = tqdm(
            total=len(task_data),
            desc=f"任務 {task_id}",
            position=task_id,
            leave=True
        )
        
        # 創建臨時輸出文件
        out_path = Path(f"data/proc/labeled_xsum_temp_{task_id}.jsonl")
        with out_path.open("w", encoding="utf-8") as out_file:
            for art in task_data:
                try:
                    sents = sent_tokenize(art["article"])
                    if not sents:  # 跳過空文章
                        task_progress.update(1)
                        continue
                    
                    # 處理文章
                    emb_s = model.encode(sents, device=device)
                    emb_h = model.encode([art["highlights"]], device=device)[0]
                    sim = (emb_s @ emb_h) / ((emb_s**2).sum(1)**0.5 * (emb_h**2).sum()**0.5)
                    
                    # 寫入結果
                    for s, score in zip(sents, sim):
                        lbl = int(score > 0.5)
                        out_file.write(json.dumps({"s": s, "y": lbl}, ensure_ascii=False) + "\n")
                    
                except Exception as e:
                    print(f"\n任務 {task_id} 處理文章時發生錯誤: {str(e)}")
                    # 繼續處理下一篇文章
                
                task_progress.update(1)
                
    except Exception as e:
        print(f"\n任務 {task_id} 發生嚴重錯誤: {str(e)}")
        raise e
    finally:
        if task_progress:
            task_progress.close()

def combine_temp_files():
    """合併所有臨時文件到最終輸出文件"""
    print("\n正在合併臨時文件...")
    success = True
    try:
        with Path("data/proc/labeled_xsum.jsonl").open("w", encoding="utf-8") as outfile:
            for i in range(1, 5):  # 1 到 4
                temp_file = Path(f"data/proc/labeled_xsum_temp_{i}.jsonl")
                if temp_file.exists():
                    try:
                        with temp_file.open("r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                        # 刪除臨時文件
                        temp_file.unlink()
                    except Exception as e:
                        print(f"合併文件 {i} 時發生錯誤: {str(e)}")
                        success = False
                else:
                    print(f"警告: 找不到臨時文件 {temp_file}")
                    success = False
    except Exception as e:
        print(f"合併過程發生錯誤: {str(e)}")
        success = False
    
    if success:
        print("合併完成！")
    else:
        print("合併過程中發生錯誤，請檢查輸出文件。")

def main():
    # 確保輸出目錄存在
    Path("data/proc").mkdir(parents=True, exist_ok=True)

    # 加載數據集
    print("正在加載數據集...")
    ds = load_dataset("json", data_files="data/raw/xsum_5pct.jsonl")["train"]
    total_articles = len(ds)
    print(f"總共加載了 {total_articles} 篇文章")
    
    # 將數據集分成4個任務
    num_tasks = 4
    chunk_size = total_articles // num_tasks + (1 if total_articles % num_tasks else 0)
    tasks = [
        list(ds.select(range(i, min(i + chunk_size, total_articles))))
        for i in range(0, total_articles, chunk_size)
    ]
    print(f"數據已分成 {len(tasks)} 個任務，每個任務約 {len(tasks[0])} 篇文章")
    
    print("\n開始處理...")
    # 使用線程池處理所有任務
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        # 提交所有任務
        future_to_task = {
            executor.submit(process_task, task_articles, task_id + 1): task_id
            for task_id, task_articles in enumerate(tasks)
        }
        
        # 等待所有任務完成
        all_success = True
        for future in future_to_task:
            try:
                future.result()
            except Exception as e:
                task_id = future_to_task[future] + 1
                print(f"\n任務 {task_id} 失敗: {e}")
                all_success = False
        
        if not all_success:
            print("\n警告：某些任務失敗了，但我們仍然會嘗試合併可用的結果")
    
    # 合併所有臨時文件
    combine_temp_files()

if __name__ == "__main__":
    main() 