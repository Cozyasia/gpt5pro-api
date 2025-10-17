# -*- coding: utf-8 -*-
"""
GPT-5 PRO API + Мини-аппа оплаты (ЮKassa)
-----------------------------------------
Содержит:
- /chat           — основной API GPT-5 PRO
- /mini           — страница тарифа (Telegram WebApp)
- /api/create-invoice — генерация ссылки оплаты ЮKassa (через Telegram Payments)
"""

import os, json, httpx
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

# ==================== CONFIG ====================
APP_NAME    = "GPT-5 PRO API"
APP_VERSION = "1.2.0"

# --- GPT config ---
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini").strip()
CORS_RAW        = os.getenv("CORS_ORIGINS", "").strip()

# --- Telegram Payments config ---
BOT_TOKEN      = os.getenv("BOT_TOKEN", "").strip()
PROVIDER_TOKEN = os.getenv("PROVIDER_TOKEN", "").strip()
BOT_USERNAME   = os.getenv("BOT_USERNAME", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("ENV OPENAI_API_KEY is required")

def parse_origins(val: str):
    if not val:
        return ["*"]
    try:
        data = json.loads(val)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return [x.strip() for x in val.split(",") if x.strip()]

ORIGINS = parse_origins(CORS_RAW)

# ==================== GPT CLIENT ====================
default_headers = {}
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL or None,
    default_headers=default_headers or None
)

# ==================== FASTAPI APP ====================
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== STATIC FILES ====================
BASE = Path(__file__).parent
STATIC = BASE / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ==================== MODELS (GPT) ====================
DEFAULT_SYSTEM = (
    "Ты дружелюбный и лаконичный ассистент на русском. "
    "Отвечай по сути, структурируй списками/шагами, не выдумывай факты. "
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    image_url: Optional[str] = Field(default=None)
    system: Optional[str] = DEFAULT_SYSTEM
    temperature: Optional[float] = 0.6
    model: Optional[str] = None

class ChatResponse(BaseModel):
    ok: bool = True
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

def build_messages(body: ChatRequest):
    msgs = [{"role": "system", "content": body.system or DEFAULT_SYSTEM}]
    if body.messages:
        msgs += [{"role": m.role, "content": m.content} for m in body.messages]
    elif body.prompt:
        msgs.append({"role": "user", "content": body.prompt})
    else:
        raise HTTPException(400, "prompt or messages required")

    if body.image_url:
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Опиши изображение"},
                {"type": "image_url", "image_url": {"url": body.image_url}},
            ]
        })
    return msgs

# ==================== ROUTES ====================

@app.get("/")
def root():
    return {"ok": True, "app": APP_NAME, "version": APP_VERSION}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        msgs = build_messages(body)
        resp = client.chat.completions.create(
            model=body.model or OPENAI_MODEL,
            messages=msgs,
            temperature=body.temperature or 0.6,
        )
        choice = resp.choices[0]
        return ChatResponse(
            text=choice.message.content.strip(),
            model=body.model or OPENAI_MODEL,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=getattr(resp, "usage", None)
        )
    except Exception as e:
        print("API error:", repr(e))
        raise HTTPException(500, "openai_error")

# ==================== MINI APP (YooKassa) ====================

@app.get("/mini", response_class=HTMLResponse)
def mini_page():
    html = (STATIC / "mini.html").read_text(encoding="utf-8") if (STATIC / "mini.html").exists() else "<h1>mini.html not found</h1>"
    return HTMLResponse(html.replace("%BOT_USERNAME%", BOT_USERNAME))

@app.post("/api/create-invoice")
async def create_invoice(req: Request):
    if not BOT_TOKEN or not PROVIDER_TOKEN:
        raise HTTPException(500, "BOT_TOKEN or PROVIDER_TOKEN missing")

    data = await req.json()
    plan = data.get("plan", "month")
    prices = {"month": 99900, "quarter": 269900, "year": 899900}  # копейки

    payload = {
        "title": "GPT-5 PRO — подписка",
        "description": "Доступ ко всем PRO-функциям бота",
        "payload": f"buy_{plan}",
        "provider_token": PROVIDER_TOKEN,
        "currency": "RUB",
        "prices": [{"label": f"Подписка ({plan})", "amount": prices[plan]}],
        "provider_data": json.dumps({
            "receipt": {
                "items": [{
                    "description": f"GPT-5 PRO ({plan})",
                    "amount": {"value": f"{prices[plan]/100:.2f}", "currency": "RUB"},
                    "vat_code": 1,
                    "quantity": "1",
                    "payment_mode": "full_prepayment",
                    "payment_subject": "service"
                }]
            }
        }),
        "need_name": False,
        "need_email": False,
        "need_phone_number": False
    }

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(f"https://api.telegram.org/bot{BOT_TOKEN}/createInvoiceLink", json=payload)
        r.raise_for_status()
        return JSONResponse({"invoice_link": r.json()["result"]})
