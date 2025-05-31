# T58
API/存放後端管理檔案以及訓練好的模型

FRONT/存放前端網頁HTML

執行方式為下載好檔案後，載入環境(requirements.txt)，開啟兩終端，一終端進入最外層資料夾輸入指令 uvicorn api.main:app --reload --host 0.0.0.0 --port 8000，啟動後端，令一終端再進入front/，執行 python main.py，當成功建立後即可開啟網頁。

環境要求請參照requirements.txt
# 專案名稱（Project Title）

> 以一句話或一段短文說明此專案的主要目標／功能

---

## 一、專案簡介（Overview）
- **任務描述**：  
  - 本專案旨在實現 XXX（例如：影像分類、文字摘要、強化學習代理人……）  
  - 主要研究／實作目標：說明為何要做這個專案、解決什麼問題。

- **關鍵特色**：  
  1. 使用神經網路＋梯度提升樹（GBDT）進行模型融合。  
  2. 支援可重複的實驗流程，並附帶範例數據。  
  3. 其他亮點（例如：支援早停、可視化訓練過程等）。

---

## 二、目錄結構（Directory Structure）
```text
AI/
├── README.md
├── requirements.txt
├── api/
│   ├── main.py
│   ├── model.py
│   └── models/
|       └── README.MD
└──  front/
    ├── main.py
    ├── templates/
    |    ├── highlight.html
    |    ├── index.html
    |    └── register.html
    └── static/
         ├── CH.png
         ├── script.js
         └── style.css 
