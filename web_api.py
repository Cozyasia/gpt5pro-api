# -*- coding: utf-8 -*-
"""
GPT-5 PRO API (с мини-аппой и инвойсами)
- POST /chat          : прокси к LLM (как было)
- GET  /              : метаданные
- GET  /health        : ok
- GET  /mini          : страница тарифов (static/mini.html)
- POST /api/create-invoice : создать invoice-ссылку через Bot API (YooKassa)
"""

import os, json
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME    = "GPT-5 PRO API"
APP_VERSION = "1.2.0"

# -------- ENV (LLM) --------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()  # напр. https://openrouter.ai/api/v1
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini").strip()
CORS_RAW        = os.getenv("CORS_ORIGINS", "").strip()

# Вежливые заголовки для OpenRouter (опционально)
OR_SITE  = os.getenv("OPENROUTER_SITE_URL", "").strip()
OR_TITLE = os.getenv("OPENROUTER_APP_NAME", "").strip()

# -------- ENV (Telegram Payments) --------
BOT_TOKEN        = os.getenv("BOT_TOKEN", "").strip()
PROVIDER_TOKEN   = os.getenv("PROVIDER_TOKEN_YOOKASSA", os.getenv("PROVIDER_TOKEN", "")).strip()
BOT_USERNAME     = os.getenv("BOT_USERNAME", "").strip()  # без @, для fallback-ссылок

# тарифы (рубли)
PLANS: Dict[str, Dict[str, Any]] = {
    "month":   {"rub": 999,  "label": "Месяц (30 дней)"},
    "quarter": {"rub": 2699, "label": "3 месяца"},
    "year":    {"rub": 8999, "label": "Год"},
}

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

# -------- LLM CLIENT --------
default_headers: Dict[str, str] = {}
if OR_SITE:  default_headers["HTTP-Referer"] = OR_SITE
if OR_TITLE: default_headers["X-Title"] = OR_TITLE

oai = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL or None,
    default_headers=default_headers or None,
)

# -------- FASTAPI APP --------
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- статика для мини-аппы ----
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ========== МОДЕЛИ ДЛЯ /chat ==========
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
    model: Optional[str] = None

class ChatResponse(BaseModel):
    ok: bool = True
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

def build_messages(body: ChatRequest) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": (body.system or DEFAULT_SYSTEM)}]
    if body.messages and len(body.messages) > 0:
        msgs += [{"role": m.role, "content": m.content} for m in body.messages]
    else:
        txt = (body.prompt or "").strip()
        if not txt:
            raise HTTPException(400, "prompt or messages required")
        msgs.append({"role": "user", "content": txt})

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

# ========== ROUTES: базовые ==========
@app.get("/")
def root():
    return {
        "ok": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
        "cors": ORIGINS,
        "payments": {
            "bot": bool(BOT_TOKEN),
            "provider": bool(PROVIDER_TOKEN),
            "username": BOT_USERNAME or None
        }
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        msgs = build_messages(body)
        model = (body.model or OPENAI_MODEL).strip()
        resp = oai.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=body.temperature or 0.6,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        usage = None
        try:
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
        print("API error:", repr(e))
        raise HTTPException(500, "openai_error")

# ========== ROUTES: мини-аппа и инвойсы ==========
@app.get("/mini", response_class=HTMLResponse)
def mini_page():
    """
    Отдаём готовый HTML из static/mini.html и подставляем имя бота.
    """
    html_file = STATIC_DIR / "mini.html"
    if not html_file.exists():
        return HTMLResponse("<h3>mini.html не найден</h3>", status_code=404)
    html = html_file.read_text(encoding="utf-8")
    return HTMLResponse(html.replace("%BOT_USERNAME%", BOT_USERNAME or ""))

@app.post("/api/create-invoice")
async def create_invoice(req: Request):
    """
    Возвращает invoice_link через Bot API createInvoiceLink.
    Нужны ENV: BOT_TOKEN, PROVIDER_TOKEN_YOOKASSA (или PROVIDER_TOKEN).
    """
    if not BOT_TOKEN or not PROVIDER_TOKEN:
        return JSONResponse(
            {"ok": False, "error": "missing_bot_or_provider_token"},
            status_code=500
        )
    body = await req.json()
    plan = (body.get("plan") or "month").lower()
    if plan not in PLANS:
        plan = "month"

    rub = int(PLANS[plan]["rub"])
    amount_kopecks = rub * 100

    # provider_data c чеком для ЮKassa — максимально «безошибочный» формат
    provider_data = {
        "receipt": {
            "items": [{
                "description": f"GPT-5 PRO — {PLANS[plan]['label']}",
                "quantity": "1.00",
                "amount": {"value": f"{rub:.2f}", "currency": "RUB"},
                "vat_code": 1,
                "payment_mode": "full_prepayment",
                "payment_subject": "service"
            }]
        }
    }

    payload = {
        "title": "Подписка GPT-5 PRO",
        "description": f"Доступ к GPT-5 PRO — {PLANS[plan]['label']}",
        "payload": f"buy_{plan}",
        "provider_token": PROVIDER_TOKEN,
        "currency": "RUB",
        "prices": [{"label": f"{PLANS[plan]['label']}", "amount": amount_kopecks}],
        "provider_data": json.dumps(provider_data),
        "need_name": False, "need_email": False, "need_phone_number": False,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/createInvoiceLink",
                json=payload
            )
        data = r.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"http_error: {e}"}, status_code=502)

    if isinstance(data, dict) and data.get("ok") and "result" in data:
        return JSONResponse({"ok": True, "invoice_link": data["result"]})

    # вернём подробности ошибки Бота
    return JSONResponse({"ok": False, "error": data}, status_code=400)
