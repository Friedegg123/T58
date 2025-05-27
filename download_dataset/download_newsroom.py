from pathlib import Path
import json
import os
from datasets import load_dataset
from tqdm.auto import tqdm
import random

def main():
    # 確保輸出目錄存在
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    print("正在下載 Newsroom 數據集...")
    
    # 使用 Hugging Face datasets 加載數據集
    print("正在加載訓練集...")
    ds = load_dataset("lil-lab/newsroom", split="train")
    
    # 計算5%的樣本大小
    sample_size = int(len(ds) * 0.05)
    print(f"總數據集大小: {len(ds)}")
    print(f"5%樣本大小: {sample_size}")
    
    # 隨機選擇5%的樣本
    indices = random.sample(range(len(ds)), sample_size)
    sampled_ds = ds.select(indices)
    
    # 將數據集轉換為所需格式並保存
    output_data = []
    
    for item in tqdm(sampled_ds, desc="處理數據"):
        # Newsroom 數據集中的字段：
        # - text: 原文
        # - summary: 摘要
        article = {
            "article": item["text"],
            "highlights": item["summary"],
            "id": str(len(output_data))
        }
        output_data.append(article)
    
    # 保存為 jsonl 格式
    output_file = "data/raw/newsroom_5pct.jsonl"
    print(f"\n正在保存到 {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    total_articles = len(output_data)
    print(f"總共保存了 {total_articles} 篇文章")

if __name__ == "__main__":
    main() 