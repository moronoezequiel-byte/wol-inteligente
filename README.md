# WOL Inteligente v8

Buscador y asistente de investigación que consulta exclusivamente artículos de la Biblioteca en línea Watchtower (WOL).

## Qué cambia en v8
- La interfaz muestra directamente los artículos; no muestra la interpretación ni las consultas internas.
- La IA crea un plan de búsqueda con consultas diferentes y complementarias.
- Consulta WOL directamente y elimina enlaces duplicados.
- Lee temporalmente fragmentos de artículos de WOL para evaluar pertinencia.
- Evalúa qué aspectos de la pregunta están cubiertos y puede hacer una segunda ronda de búsquedas para cubrir aspectos faltantes.
- Presenta únicamente títulos, datos bibliográficos que WOL permita detectar y enlaces directos a los artículos originales.
- No genera resúmenes ni respuestas doctrinales.

## Instalación en Render
Build command:
```bash
pip install -r requirements.txt
```

Start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Variables de entorno:
- `OPENAI_API_KEY`: configurar como variable privada en Render. Nunca subirla a GitHub.
- `OPENAI_MODEL`: modelo disponible para tu API. Si ya está configurado en Render, se conserva.

Health check: `/health`

## Importante
- El texto de WOL se procesa temporalmente para evaluar relevancia. No se guarda en una base de datos ni se envía a la interfaz.
- La aplicación muestra enlaces a los artículos originales; no reproduce el contenido completo.
- La extracción depende de la estructura y disponibilidad de WOL. Si WOL cambia su sitio o limita solicitudes, habrá que adaptar el extractor.
- Si la API de IA no está disponible, se usa un modo local de respaldo menos preciso.
- La versión no se considera validada hasta probarla desplegada contra WOL y la API configurada.
