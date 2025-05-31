# Highlight Agent

> 自動為使用者提供文章重點句子，並支援基於使用者回饋的個性化微調

---

## 一、專案簡介（Overview）

- **任務描述**  
  - 本專案旨在實作一套「Highlight Agent」系統：  
    1. 使用者可以在網頁上貼入文章 URL 或全文文字，系統會自動分析並找出文章中的重點句子，並以高亮方式呈現。  
    2. 使用者能對系統標註出的重點句子給予「好／壞」回饋，後端會收集這些二元標籤，並定期（或按需）將回饋資料整合到模型重新訓練流程，漸進式微調模型以符合個人需求。  

- **主要目標**  
  1. **自動化重點抽取**：從原始文章中分句、計算句子嵌入、透過輕量級分類器（MLP + 梯度提升樹）判斷每句是否為重點。  
  2. **使用者回饋迴圈**：提供一個簡易的 GUI 介面讓使用者可對系統的「高亮建議」打分，這些回饋將作為新的訓練標籤，讓模型能針對不同使用者偏好做微調。  
  3. **輕量化部署**：後端使用 FastAPI + PyTorch 與 scikit-learn/CatBoost/XGBoost，前端則以基本 HTML/CSS/JavaScript 呈現。即便是在無 GPU 的普通伺服器上，也能流暢地提供高亮建議。  

- **關鍵特色**  
  1. **SBERT 嵌入 + MLP/GBDT 混合模型**  
     - 先利用 [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)為每句一句產生 768 維向量。  
     - 再結合多層感知機 (MLP) + CatBoost + XGBoost 組成一個三模型集成 (ensemble) 預測架構，確保對正負樣本不平衡資料能有較佳表現。  
  2. **使用者回饋微調**  
     - 前端提供「讚／踩」按鈕，使用者可以二元標註系統高亮的句子是否符合需求。  
     - 回饋資料會存進資料庫（或 JSONL 檔），透過離線批次訓練或每日自動化 DAG，將新標籤整合到下次模型訓練。  
     - 最終可依照個人評分習慣產生「個人專屬」的微調模型。  
  3. **極簡前後端設計**  
     - 後端採用 FastAPI 提供 RESTful API：包含「文章文字擷取」、「句子嵌入編碼」、「高亮預測」「回饋收集」四大端點；  
     - 前端以靜態 HTML + JavaScript 實現，可以嵌入到任何網頁。未來可輕鬆擴充成 Chrome Extension 或者 Next.js App。  
  4. **離線開發 & 本地部署**  
     - 全部關鍵步驟（分句、嵌入、訓練、推論）都可以在本地端完成，不強制必須上雲。  
     - 提供範例資料集與腳本，確保開箱即用。  
  5. **易於擴展**  
     - 模組化設計：若未來要替換嵌入模型（如改成自己的 BERT 微調模型、或者外部 LLM API），只需要改 `encode_embeddings.py` 或者 API 呼叫邏輯即可。  

---

## 二、目錄結構（Directory Structure）

```plaintext
AI/
├── README.md
├── requirements.txt
├── api/
│   ├── main.py
│   ├── model.py
│   └── models/
│       └── README.MD
└── front/
    ├── main.py
    ├── templates/
    │    ├── highlight.html
    │    ├── index.html
    │    └── register.html
    └── static/
         ├── CH.png
         ├── script.js
         └── style.css
```

---
#小提醒
---
環境設置請參照requirements.txt
python為3.10版本
啟動後端指令   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
啟動前端指令   進入front/執行python main.py
實驗結果:
![image](https://github.com/user-attachments/assets/a13a39c4-bf24-47ba-a57d-bb388c4832bb)

