# WOL Inteligente v6 — prueba

Flujo:
Pregunta natural → IA interpreta → 3 búsquedas → WOL → títulos y enlaces de artículos.

La IA NO responde la pregunta religiosa. Los artículos y enlaces mostrados provienen de WOL.

## Render
Build:
pip install -r requirements.txt

Start:
uvicorn main:app --host 0.0.0.0 --port $PORT

Variables:
OPENAI_API_KEY
OPENAI_MODEL=gpt-6-luna

## Prueba
Antes de subir esta versión a Render, probar:
¿Cómo puedo ayudar a alguien que está desanimado?

Debe mostrar artículos de WOL con botón "ABRIR ARTÍCULO EN WOL".
