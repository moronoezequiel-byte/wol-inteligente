# WOL Inteligente v5.1 — Render

Aplicación FastAPI de un solo servicio: frontend + API.

## Render

Usar un Web Service conectado a este repositorio.

- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Variables de entorno:
- `OPENAI_API_KEY` = tu clave privada
- `OPENAI_MODEL` = el modelo de OpenAI que quieras utilizar

No subas una clave real a GitHub.

La aplicación sirve la interfaz en `/` y la API en `/api/search`.
