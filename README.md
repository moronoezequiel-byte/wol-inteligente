# WOL Inteligente v9

Buscador y asistente de investigación que consulta exclusivamente artículos de la Biblioteca en línea Watchtower (WOL).

## Cambios de v9
- Extracción más tolerante de enlaces de artículos de WOL, incluidos enlaces relativos y distintos contenedores HTML.
- Registros técnicos en Render para distinguir errores HTTP, páginas recibidas sin enlaces reconocibles y cantidad de candidatos encontrados.
- Más consultas complementarias para temas de desánimo, vejez y edad avanzada.
- Evaluación con respaldo local: si la IA descarta todos los candidatos, se recuperan artículos pertinentes mediante coincidencias ampliadas; también se agregan unos pocos candidatos locales cuando ayudan a mejorar la cobertura.
- Se revisan hasta 30 artículos únicos en la primera etapa.
- La interfaz conserva el diseño sencillo de la versión anterior y muestra únicamente títulos, datos bibliográficos disponibles y enlaces directos a WOL.
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

Health check: `/health` (debe informar versión `9.0`).

## Prueba recomendada
Después de desplegar, buscar exactamente: **Cómo afrontar el desánimo en la vejez**. WOL tiene artículos conocidos sobre la alegría en la vejez y las dificultades de la edad avanzada; esta consulta sirve como prueba de regresión.

## Privacidad y alcance
- El texto de WOL se procesa temporalmente para evaluar relevancia. No se guarda en una base de datos ni se envía a la interfaz.
- La aplicación muestra enlaces a los artículos originales; no reproduce el contenido completo.
- La versión debe probarse desplegada contra WOL y la API configurada antes de considerarse validada.
