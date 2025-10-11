# -*- coding: utf-8 -*-
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME = "GPT-5 PRO API"
APP_VERSION = "0.1.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("ENV OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# ---- CORS (укажите домен вашего веб-приложения через ENV CORS_ORIGINS) ----
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if not origins:
    # по умолчанию разрешим всё, чтобы не ловить 403 во время настройки
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "Ты дружелюбный и лаконичный ассистент на русском. "
    "Отвечай по сути, структурируй списками/шагами, не выдумывай факты. "
    "Если ссылаешься на источники — в конце дай короткий список ссылок."
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    # Вариант 1 (простой): prompt
    prompt: Optional[str] = None
    # Вариант 2 (диалог): messages
    messages: Optional[List[Message]] = None
    # Опционально: URL изображения для vision
    image_url: Optional[str] = Field(default=None, description="URL или data: URI")

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME, "version": APP_VERSION}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        if req.messages:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in req.messages:
                msgs.append({"role": m.role, "content": m.content})
        else:
            text = req.prompt or ""
            if not text.strip():
                raise HTTPException(status_code=400, detail="prompt or messages required")
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}]

        # Vision? добавим картинку в последний юзер-сообщение
        if req.image_url:
            # найдём последнее user-сообщение или создадим новое
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "user":
                    # заменим content на массив "мультимодальности"
                    user_text = msgs[i]["content"] if isinstance(msgs[i]["content"], str) else ""
                    msgs[i]["content"] = [
                        {"type": "text", "text": user_text or "Опиши изображение и извлеки важные детали."},
                        {"type": "image_url", "image_url": {"url": req.image_url}}
                    ]
                    break
            else:
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Опиши изображение."},
                        {"type": "image_url", "image_url": {"url": req.image_url}}
                    ]
                })

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.6,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return ChatResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        # лог в stdout — Render покажет в логах
        print("API error:", repr(e))
        raise HTTPException(status_code=500, detail="openai_error")

# Для локального запуска: uvicorn web_api:app --reload
