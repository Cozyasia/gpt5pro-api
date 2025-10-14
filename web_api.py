# -*- coding: utf-8 -*-
"""
GPT-5 PRO API
- POST /chat:  prompt ИЛИ messages, опционально image_url (vision), system, temperature, model
- GET  /      : метаданные сервиса (убирает 404 на health-checkи)
- GET  /health: ok
Работает и с OpenAI, и с OpenRouter (задаётся OPENAI_BASE_URL)
CORS_ORIGINS: JSON-массив или CSV-строка доменов.
"""
import os, json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME    = "GPT-5 PRO API"
APP_VERSION = "1.1.0"

# -------- ENV --------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()  # для OpenRouter: https://openrouter.ai/api/v1
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini").strip()  # с OpenAI можно "gpt-4o-mini"
CORS_RAW        = os.getenv("CORS_ORIGINS", "").strip()

# Вежливые заголовки для OpenRouter (опционально, но рекомендуется)
OR_SITE  = os.getenv("OPENROUTER_SITE_URL", "").strip()
OR_TITLE = os.getenv("OPENROUTER_APP_NAME", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("ENV OPENAI_API_KEY is required")

def parse_origins(val: str) -> List[str]:
    if not val:
        return ["*"]
    try:
        lst = json.loads(val)
        if isinstance(lst, list) and all(isinstance(x, str) for x in lst):
            return lst
    except Exception:
        pass
    return [x.strip() for x in val.split(",") if x.strip()]

ORIGINS = parse_origins(CORS_RAW)

# -------- CLIENT --------
default_headers: Dict[str, str] = {}
if OR_SITE:  default_headers["HTTP-Referer"] = OR_SITE
if OR_TITLE: default_headers["X-Title"] = OR_TITLE

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL or None,       # пусто → api.openai.com/v1
    default_headers=default_headers or None # для OpenRouter
)

# -------- APP --------
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODELS --------
DEFAULT_SYSTEM = (
    "Ты дружелюбный и лаконичный ассистент на русском. "
    "Отвечай по сути, структурируй списками/шагами, не выдумывай факты. "
    "Если ссылаешься на источники — в конце дай короткий список ссылок."
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    image_url: Optional[str] = Field(default=None, description="URL или data:URI (data:image/png;base64,...)")
    system: Optional[str] = DEFAULT_SYSTEM
    temperature: Optional[float] = 0.6
    model: Optional[str] = None  # можно переопределить модель на запрос

class ChatResponse(BaseModel):
    ok: bool = True
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # prompt_tokens / completion_tokens / total_tokens

# -------- HELPERS --------
def build_messages(body: ChatRequest) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": (body.system or DEFAULT_SYSTEM)}]
    if body.messages and len(body.messages) > 0:
        msgs += [{"role": m.role, "content": m.content} for m in body.messages]
    else:
        txt = (body.prompt or "").strip()
        if not txt:
            raise HTTPException(400, "prompt or messages required")
        msgs.append({"role": "user", "content": txt})

    # vision: прикрепим картинку к последнему user-сообщению
    if body.image_url:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                user_text = msgs[i]["content"] if isinstance(msgs[i]["content"], str) else ""
                msgs[i]["content"] = [
                    {"type": "text", "text": user_text or "Опиши изображение и извлеки важные детали."},
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
    return msgs

# -------- ROUTES --------
@app.get("/")
def root():
    return {
        "ok": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
        "cors": ORIGINS,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        msgs = build_messages(body)
        model = (body.model or OPENAI_MODEL).strip()

        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=body.temperature or 0.6,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        usage = None
        try:
            # OpenAI SDK v1+ / OpenRouter: структура схожа
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        except Exception:
            pass

        return ChatResponse(
            text=answer,
            model=model,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        # покажем в логах Render
        print("API error:", repr(e))
        raise HTTPException(500, "openai_error")

# Для локального запуска:
# uvicorn web_api:app --reload
