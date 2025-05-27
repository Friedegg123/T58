from datasets import load_dataset
import json
from pathlib import Path

# 創建輸出目錄
Path("data/raw").mkdir(parents=True, exist_ok=True)

# 下載數據集
dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:5%]", trust_remote_code=True)

# 保存為 JSONL 格式
output_file = Path("data/raw/cnndm_5pct.jsonl")
with output_file.open("w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"數據集已下載並保存到 {output_file}") 