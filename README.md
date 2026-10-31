# WOL Inteligente v7

Buscador y asistente de investigación que consulta exclusivamente Watchtower ONLINE LIBRARY (WOL).

## Qué hace
- Interpreta la necesidad expresada en lenguaje natural.
- Genera de 4 a 6 búsquedas distintas y complementarias.
- Consulta WOL directamente en español.
- Abre y lee temporalmente fragmentos de los artículos encontrados para evaluar su pertinencia.
- Usa IA para seleccionar artículos pertinentes y etiquetar qué necesidad cubren.
- Muestra únicamente títulos, datos bibliográficos disponibles y enlaces directos a los artículos originales. No genera resúmenes ni respuestas doctrinales.

El contenido de los artículos se procesa en memoria para evaluar relevancia; no se guarda en una base de datos ni se muestra en la interfaz.

## Render
Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:
- `OPENAI_API_KEY`: clave privada de API (configurar solo en Render, nunca en GitHub).
- `OPENAI_MODEL`: modelo habilitado para tu cuenta/API. Si no se define, el código usa `gpt-6`.

Health check: `/health`

## Notas
- La consulta a WOL depende de la disponibilidad y del HTML que entregue WOL. Si cambia su estructura, puede ser necesario ajustar el extractor.
- El servicio no copia ni redistribuye el corpus de WOL; muestra enlaces a los artículos originales.
- El ranking automático solo puede evaluar el texto que el extractor consiga leer. Los resultados deben verificarse abriendo el artículo original.
