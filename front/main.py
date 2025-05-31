from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from newspaper import Article
import nltk, re, uvicorn
from passlib.context import CryptContext
from datetime import datetime
import os
# nltk.download("punkt") # Can be commented out if already downloaded and causing issues on startup

# =============================
# 資料庫設定
# =============================
DB_USER = "root"
DB_PASSWORD = "YourNewRootPassword"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "AI"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# =============================
# 密碼加密設定
# =============================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

# =============================
# 資料表定義
# =============================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)

    feedbacks = relationship("UserFeedback", back_populates="user")


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    feedback = Column(Integer, nullable=False)
    sentence_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")

class LoginRequest(BaseModel):
    username: str
    password: str
    
class LoginData(BaseModel):
    username: str
    password: str
    
class FeedbackData(BaseModel):
    label: int
    user_id: int
    sentence_score: float

class UserThresholdResponse(BaseModel):
    threshold: float

# 建立資料表
Base.metadata.create_all(bind=engine)

# =============================
# FastAPI App 設定
# =============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 若前端部署後記得換成你的 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載 static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 取得資料庫 Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================
# 頁面路由
# =============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/index.html", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/register.html", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/highlight.html", response_class=HTMLResponse)
async def highlight_page(request: Request):
    return templates.TemplateResponse("highlight.html", {"request": request})

# =============================
# 使用者註冊 API
# =============================

@app.post("/register")
def register_user(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter_by(username=username).first()
    if existing_user:
        return {"success": False, "message": "User name already exists"}

    hashed = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed)
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "Sign up successfully"}
#登入

@app.post("/login")
def login_user(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    if not pwd_context.verify(data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Password error")
    return {"success": True,
            "user_id": user.id,
            "username": user.username,
            "access_token": "dummy-token"
            }
    
@app.post("/feedback")
def save_feedback(data: FeedbackData, db: Session = Depends(get_db)):
    feedback_entry = UserFeedback(
        user_id = data.user_id,
        feedback = data.label,
        sentence_score = data.sentence_score
    )
    db.add(feedback_entry)
    try:
        db.commit()
        return {"success": True}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Duplicate or save failed")

@app.get("/api/user-threshold/{user_id}", response_model=UserThresholdResponse)
async def get_user_threshold(user_id: int, db: Session = Depends(get_db)):
    feedback_records = db.query(UserFeedback.sentence_score, UserFeedback.feedback)\
        .filter(UserFeedback.user_id == user_id).all()

    # Default values from highlight.html slider
    default_threshold = 0.27
    min_slider_threshold = 0.17
    max_slider_threshold = 0.37

    if not feedback_records:
        return UserThresholdResponse(threshold=default_threshold)

    liked_scores = [score for score, fb_val in feedback_records if fb_val == 1 and score is not None]
    disliked_scores = [score for score, fb_val in feedback_records if fb_val == 0 and score is not None]

    calculated_threshold = default_threshold
    has_liked = bool(liked_scores)
    has_disliked = bool(disliked_scores)

    if has_liked and has_disliked:
        avg_liked = sum(liked_scores) / len(liked_scores)
        avg_disliked = sum(disliked_scores) / len(disliked_scores)
        if avg_liked > avg_disliked:
            calculated_threshold = (avg_liked + avg_disliked) / 2.0
        else:
            calculated_threshold = avg_liked # Prioritize showing what they liked
    elif has_liked:
        avg_liked = sum(liked_scores) / len(liked_scores)
        calculated_threshold = avg_liked 
    elif has_disliked:
        avg_disliked = sum(disliked_scores) / len(disliked_scores)
        calculated_threshold = avg_disliked
    
    # Clamp to slider's visible range and round
    final_threshold = max(min_slider_threshold, min(calculated_threshold, max_slider_threshold))
    final_threshold = round(final_threshold, 2) # Round to 2 decimal places for slider step

    return UserThresholdResponse(threshold=final_threshold)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
    
