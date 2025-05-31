import datasets
from datasets import load_dataset
import random
import json
import os

def download_cnn_dailymail(sample_ratio=0.25):
    """
    下載 CNN/DailyMail 數據集並保存指定比例的樣本
    
    Args:
        sample_ratio (float): 要保存的數據集比例 (0.25 = 25%)
    """
    print(f"正在下載 CNN/DailyMail 數據集 ({sample_ratio*100}% 的數據)...")
    
    # 載入數據集
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    
    # 為每個分割創建目錄
    os.makedirs("data", exist_ok=True)
    
    # 處理每個分割
    for split in ['train', 'validation', 'test']:
        print(f"\n處理 {split} 集...")
        current_dataset = dataset[split]
        
        # 計算要採樣的數量
        sample_size = int(len(current_dataset) * sample_ratio)
        
        # 隨機採樣
        indices = random.sample(range(len(current_dataset)), sample_size)
        sampled_data = [current_dataset[i] for i in indices]
        
        # 保存數據
        output_file = f"data/cnn_dailymail_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(sampled_data)} 條記錄到 {output_file}")

if __name__ == "__main__":
    # 下載並保存 25% 的數據集
    download_cnn_dailymail(0.25)
    print("\n數據集下載完成！")