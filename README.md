# WOL Inteligente v10

La v10 convierte el buscador en un investigador general, no en un sistema construido alrededor de un tema concreto.

Para cada pregunta:
- entiende la necesidad real;
- genera un mapa interno de conceptos, sinónimos, situaciones, acciones y vocabulario probable de WOL;
- hace 8-12 búsquedas complementarias;
- reúne y lee artículos reales;
- evalúa pertinencia contra la necesidad, no contra una palabra;
- detecta huecos y puede hacer una segunda ronda de búsquedas;
- muestra solo artículos originales de WOL.

La interfaz no muestra el razonamiento interno, las consultas generadas ni resúmenes.

Render:
Build: `pip install -r requirements.txt`
Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Variables:
- `OPENAI_API_KEY`: privada.
- `OPENAI_MODEL`: conservar el modelo que ya funciona.

`/health` debe informar versión `10.0`.

Pruebas recomendadas:
- Cómo conducir el Estudio de La Atalaya.
- Cómo ayudar a alguien que está desanimado.
- Cómo preparar un discurso.
- Cómo enseñar a un hijo a orar.
- Qué hacer cuando cuesta concentrarse al estudiar.
- Preguntas sobre personajes o acontecimientos bíblicos.
- Preguntas coloquiales donde WOL probablemente use vocabulario diferente.
