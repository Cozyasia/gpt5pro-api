# GPT-5 PRO API

FastAPI-шлюз для веб-версии чата.

### Локальный старт
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn web_api:app --reload
# http://127.0.0.1:8000/health
