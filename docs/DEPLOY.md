# LogicGuard Deployment

## Local API

```bash
pip install -e ".[dev]"
python run_api.py --kb knowledge_base_1200.json --port 8000
```

Endpoints:
- `GET /docs` — Swagger UI
- `POST /api/v1/validate` — single query
- `POST /api/v1/batch` — batch (max 50)
- `POST /api/v1/kb/upload` — upload domain KB JSON
- `GET /api/v1/kb/stats` — current KB stats

## Railway

1. Set start command: `logicguard-api --host 0.0.0.0 --port $PORT`
2. Env vars: `LOGICGUARD_KB=knowledge_base_1200.json`, `LOGICGUARD_MODEL=llama3.2:3b`
3. Ollama must be reachable for LLM parser Stage 1 (regex mode works without Ollama in research scripts)

## LangChain integration

```python
from integrations.langchain_middleware import LogicGuardMiddleware

guard = LogicGuardMiddleware("knowledge_base_1200.json")
result = guard.guard("Are all dogs mammals?", llm_answer="yes")
print(result.epistemic_state, result.final_answer)
```
