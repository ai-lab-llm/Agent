from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logger import setup_logging
from app.routes import chat

setup_logging("INFO")
app = FastAPI(title="Protectee SQL Agent API", debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기존 루트 헬스
@app.get("/health")
async def health():
    return {"status": "ok"}

# /api/health 헬스도 추가
@app.get(f"{settings.api_prefix}/health")
async def api_health():
    return {"status": "ok"}

# /api/ask 등 라우터
app.include_router(chat.router, prefix=settings.api_prefix, tags=["chat"])