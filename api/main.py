from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk import sent_tokenize, download
import newspaper
import uvicorn
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List # 確保導入 List (對於 Python < 3.9)

# 導入你的評分模型模塊
from . import model as scoring_model # 使用別名以區分 SentenceTransformer model

# 載入預訓練的 sentence-transformer 模型 (用於嵌入，與評分模型分開)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model.to(device)

# 下載必要的 NLTK 資源
try:
    download('punkt')
    download('averaged_perceptron_tagger')
    # download('punkt_tab') # 如果之前添加過，確認是否仍需要
except Exception as e:
    print(f"警告：NLTK 資源下載失敗 - {str(e)}")

app = FastAPI()

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 模型定義 ---
class UrlRequest(BaseModel):
    url: str

class AutoHighlightRequest(BaseModel):
    sentences: List[str]

class AutoHighlightResponse(BaseModel):
    scores: List[float]
    threshold: float

@app.post("/api/extract-url")
async def extract_url(request: UrlRequest):
    try:
        print(f"收到請求 /api/extract-url：{request.url}")
        article = newspaper.Article(request.url)
        article.download()
        article.parse()
        
        text = article.text
        title = article.title
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="無法從該URL擷取到文章內容或內容為空")
        
        # 分句，這裡的句子是原始的，用於顯示和後續發送到 /api/auto-highlight
        raw_sentences = sent_tokenize(text)
        # 過濾掉完全是空白的句子，但不做更複雜的處理，保持原始性
        display_sentences = [s for s in raw_sentences if s.strip()]

        if not display_sentences:
             raise HTTPException(status_code=400, detail="文章內容有效，但未能分割出有效句子串列。")

        response_data = {
            "success": True,
            "data": {
                "title": title,
                "text": text, # 返回原始文本，前端的 segmentText 會處理段落和分句
                # "sentences": display_sentences, # 前端會從 'text' 自己分句
                "url": request.url
            }
        }
        # print(f"回應數據 /api/extract-url：{response_data}") # 嵌入體積大，避免打印
        print(f"回應數據 /api/extract-url 成功，標題: {title}")
        return response_data
        
    except newspaper.article.ArticleException as e:
        print(f"Newspaper 擷取錯誤 for {request.url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"文章擷取失敗 (newspaper error): {str(e)}")
    except Exception as e:
        print(f"錯誤 /api/extract-url for {request.url}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500,detail=f"處理文章擷取時發生內部錯誤: {str(e)}")

# --- 新的 /api/auto-highlight 端點 ---
@app.post("/api/auto-highlight", response_model=AutoHighlightResponse)
async def auto_highlight_endpoint(request: AutoHighlightRequest):
    print(f"收到請求 /api/auto-highlight，句子數量: {len(request.sentences)}")
    if not request.sentences:
        print("警告 /api/auto-highlight: 收到空句子列表")
        return AutoHighlightResponse(scores=[], threshold=0.5)

    try:
        # 檢查 api/model.py 中實際加載的模型組件
        if scoring_model.V_SCALER is None or scoring_model.C_CLASSIFIER is None:
            # 使用 V_SCALER 和 C_CLASSIFIER 進行檢查
            print("CRITICAL ERROR in /api/auto-highlight: 評分模型 (Scaler or Classifier from model.py) 未加載.")
            raise HTTPException(status_code=503, detail="評分模型服務暫不可用，請檢查服務器啟動日誌。")

        # 1. 將句子轉換為嵌入向量
        # 過濾掉空字符串，以避免 embedding_model 出錯
        valid_sentences = [s for s in request.sentences if s.strip()]
        if not valid_sentences:
            print("警告 /api/auto-highlight: 過濾後句子列表為空")
            return AutoHighlightResponse(scores=[], threshold=0.5)
            
        print(f"正在為 {len(valid_sentences)} 個有效句子生成嵌入向量...")
        embeddings = embedding_model.encode(
            valid_sentences,
            batch_size=32, 
            show_progress_bar=False, 
            device=device
        )
        # embedding_model.encode 返回 numpy 數組列表，需要轉換為 list of lists or list of np.arrays
        # score 函數期望 a list of embedding arrays
        embeddings_list = [emb for emb in embeddings] 
        print(f"嵌入向量生成完畢，數量: {len(embeddings_list)}")

        # 2. 使用嵌入向量列表調用 scoring_model.score
        predicted_scores = scoring_model.score(embeddings_list)
        
        calculated_threshold = 0.41 # 保持簡單

        print(f"/api/auto-highlight 處理完畢. 返回分數數量: {len(predicted_scores)}, 閾值: {calculated_threshold}")
        return AutoHighlightResponse(scores=predicted_scores, threshold=calculated_threshold)

    except AttributeError as e:
        print(f"ERROR in /api/auto-highlight (AttributeError, model components might be None): {e}")
        raise HTTPException(status_code=500, detail=f"處理高亮請求時發生屬性錯誤（評分模型組件可能未正確加載）： {str(e)}")
    except Exception as e:
        print(f"UNEXPECTED ERROR in /api/auto-highlight: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"處理自動高亮請求時發生內部錯誤: {str(e)}")

if __name__ == "__main__":
    print("啟動後端 API 服務在 http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 