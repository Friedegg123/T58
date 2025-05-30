from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from newspaper import Article
import nltk, re
from passlib.context import CryptContext
from datetime import datetime
import os
nltk.download("punkt")

# =============================
# 資料庫設定
# =============================
DB_USER = "root"
DB_PASSWORD = ""
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
    sentence_id = Column(String(128), nullable=False)
    label_type = Column(String(32), nullable=False)
    feedback = Column(String(8), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")

    __table_args__ = (UniqueConstraint('user_id', 'sentence_id', name='uix_user_sentence'),)
    
class LoginRequest(BaseModel):
    username: str
    password: str
    
class LoginData(BaseModel):
    username: str
    password: str
    
class FeedbackData(BaseModel):
    sentence: str
    label: int
    user_id: int


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
        return {"success": False, "message": "使用者名稱已存在"}

    hashed = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed)
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "註冊成功"}
#登入

@app.post("/login")
def login_user(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="使用者不存在")
    if not pwd_context.verify(data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="密碼錯誤")
    return {"success": True,
            "user_id": user.id,
            "username": user.username,
            "access_token": "dummy-token"
            }
    
@app.post("/feedback")
def save_feedback(data: FeedbackData, db: Session = Depends(get_db)):
    feedback = UserFeedback(
        user_id = data.user_id,
        sentence_id = data.sentence,
        label_type = "highlight",
        feedback = str(data.label)
    )
    db.add(feedback)
    try:
        db.commit()
        return {"success": True}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="重複或儲存失敗")
    
