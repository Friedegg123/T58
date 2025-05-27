from pathlib import Path
import json
import os
from datasets import load_dataset
from tqdm.auto import tqdm

def main():
    # 確保輸出目錄存在
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    print("正在下載 BBC 新聞數據集...")
    
    # 使用 Hugging Face datasets 加載數據集
    ds = load_dataset("SetFit/bbc-news")
    
    # 將數據集轉換為所需格式並保存
    output_data = []
    
    for item in tqdm(ds["train"], desc="處理數據"):
        article = {
            "article": item["text"],
            "highlights": item["label_text"],
            "id": str(len(output_data))
        }
        output_data.append(article)
    
    # 保存為 jsonl 格式
    output_file = "data/raw/bbc_5pct.jsonl"
    print(f"\n正在保存到 {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    total_articles = len(output_data)
    print(f"總共保存了 {total_articles} 篇文章")

if __name__ == "__main__":
    main() 