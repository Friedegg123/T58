from pathlib import Path
import json
import os
from datasets import load_dataset
from tqdm.auto import tqdm
import random

def main():
    # 確保輸出目錄存在
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    print("正在下載 XSum 數據集...")
    
    # 使用 Hugging Face datasets 加載數據集
    ds = load_dataset("xsum")
    
    # 計算5%的樣本大小
    sample_size = int(len(ds["train"]) * 0.05)
    
    # 隨機選擇5%的樣本
    indices = random.sample(range(len(ds["train"])), sample_size)
    sampled_ds = ds["train"].select(indices)
    
    # 將數據集轉換為所需格式並保存
    output_data = []
    
    for item in tqdm(sampled_ds, desc="處理數據"):
        article = {
            "article": item["document"],
            "highlights": item["summary"],
            "id": str(len(output_data))
        }
        output_data.append(article)
    
    # 保存為 jsonl 格式
    output_file = "data/raw/xsum_5pct.jsonl"
    print(f"\n正在保存到 {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    total_articles = len(output_data)
    print(f"總共保存了 {total_articles} 篇文章")

if __name__ == "__main__":
    main() 