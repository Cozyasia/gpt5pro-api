# -*- coding: utf-8 -*-
"""
GPT-5 PRO API — единый эндпоинт /chat:
- prompt ИЛИ messages
- опционально image_url (vision)
- настраиваемые system/temperature
- аккуратный CORS (JSON-массив ИЛИ CSV)
"""
import os, json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_NAME    = "GPT-5 PRO API"
APP_VERSION = "1.0.0"

# -------- ENV --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
CORS_RAW       = os.getenv("CORS_ORIGINS", "").strip()

def parse_origins(val: str) -> List[str]:
    """Принимает JSON-массив или CSV-строку доменов."""
    if not val:
        return ["*"]  # на старте пусть работает; позже лучше сузить домены
    try:
        lst = json.loads(val)
        if isinstance(lst, list) and all(isinstance(x, str) for x in lst):
            return lst
    except Exception:
        pass
    # CSV
    return [x.strip() for x in val.split(",") if x.strip()]

ORIGINS = parse_origins(CORS_RAW)

# -------- APP --------
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------- MODELS --------
DEFAULT_SYSTEM = (
    "Ты дружелюбный и лаконичный ассистент на русском. "
    "Отвечай по сути, структурируй шагами/списками, не выдумывай факты. "
    "Если даёшь источники — добавь в конце короткий список ссылок."
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: Optional[str] = None          # простой режим
    messages: Optional[List[Message]] = None  # диалог
    image_url: Optional[str] = None       # vision: URL или data:URI (data:image/png;base64,...)
    system: Optional[str] = DEFAULT_SYSTEM
    temperature: Optional[float] = 0.6

class ChatResponse(BaseModel):
    ok: bool = True
    text: str

# -------- ROUTES --------
@app.get("/health")
def health():
    return {
        "ok": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "model": OPENAI_MODEL,
        "cors": ORIGINS,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if not client:
        raise HTTPException(500, "OPENAI_API_KEY is missing")

    # Собираем messages
    msgs: List[dict] = [{"role": "system", "content": (body.system or DEFAULT_SYSTEM)}]

    if body.messages and len(body.messages) > 0:
        # диалоговый режим
        for m in body.messages:
            msgs.append({"role": m.role, "content": m.content})
    else:
        # простой prompt
        txt = (body.prompt or "").strip()
        if not txt:
            raise HTTPException(400, "prompt or messages required")
        msgs.append({"role": "user", "content": txt})

    # Vision: прикрепить image_url к последнему user-сообщению
    if body.image_url:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                last_text = msgs[i]["content"] if isinstance(msgs[i]["content"], str) else ""
                msgs[i]["content"] = [
                    {"type": "text", "text": last_text or "Опиши изображение и извлеки важные детали."},
                    {"type": "image_url", "image_url": {"url": body.image_url}},
                ]
                break
        else:
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши изображение и извлеки важные детали."},
                    {"type": "image_url", "image_url": {"url": body.image_url}},
                ]
            })

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=body.temperature or 0.6,
        )
        text = (resp.choices[0].message.content or "").strip()
        return ChatResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        # Render покажет в логах
        print("API error:", repr(e))
        raise HTTPException(500, "openai_error")
