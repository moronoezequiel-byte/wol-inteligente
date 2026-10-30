
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
app = FastAPI(title="WOL Inteligente v5")

class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    filter: str = ""

SYSTEM_PROMPT = """
Sos el intérprete semántico de WOL Inteligente.

Tu única función es transformar una pregunta natural del usuario en una estrategia
de búsqueda para la Biblioteca en línea Watchtower (WOL). NO respondas la pregunta.
NO redactes doctrina. NO cites fuentes externas. NO inventes contenido de WOL.

Interpretá:
- qué intenta encontrar la persona;
- el tema concreto;
- la situación o contexto;
- el objetivo de la búsqueda;
- conceptos y expresiones que probablemente aparezcan en WOL.

Generá exactamente 3 consultas para WOL. Deben ser semánticamente distintas:
1) la expresión principal del tema;
2) una formulación con sinónimos o términos relacionados;
3) una formulación orientada a la situación/objetivo.

Las consultas deben ser concisas, naturales y útiles para una búsqueda textual
en WOL. No las conviertas en preguntas largas. No agregues explicaciones dentro
de las consultas.

Si la pregunta usa términos coloquiales, traducilos a conceptos de búsqueda
más precisos. Por ejemplo, "¿cómo hago para que la Atalaya no sea aburrida?"
debería orientarse hacia conductor del Estudio de La Atalaya, participación,
enseñanza, preparación o hacer interesante el estudio; no limitarse a buscar
"aburrida".

La salida debe contener:
intent, topic, context, goal, concepts y queries.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "topic": {"type": "string"},
        "context": {"type": "string"},
        "goal": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 12},
        "queries": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
    },
    "required": ["intent", "topic", "context", "goal", "concepts", "queries"],
    "additionalProperties": False,
}

def local_fallback(q: str):
    # Respaldo explícito: NO pretende ser IA. Solo mantiene la app funcional.
    s = q.lower()
    concepts = []
    groups = [
        (["dinero", "plata", "poco dinero", "económ", "deuda", "gastos"],
         ["administrar el dinero", "presupuesto", "gastos", "deudas", "ahorrar", "vivir con menos"]),
        (["atalaya", "conducir", "presidir", "coment", "estudio"],
         ["conductor del Estudio de La Atalaya", "presidir el estudio", "preparación", "participación", "enseñanza"]),
        (["desanim", "triste", "bajone", "sin ganas"],
         ["desánimo", "animar", "consolar", "fortalecer", "ayudar a otros"]),
        (["gog", "magog"],
         ["Gog de Magog", "Ezequiel 38", "Ezequiel 39", "ataque de Gog", "pueblo de Dios"]),
    ]
    for keys, vals in groups:
        if any(k in s for k in keys):
            concepts.extend(vals)
    if not concepts:
        words = [w.strip("¿?¡!.,;:") for w in q.split() if len(w) > 4]
        concepts = words[:8] or ["información", "consejos", "Biblia"]
    concepts = list(dict.fromkeys(concepts))
    return {
        "intent": "Encontrar información relacionada con la pregunta",
        "topic": ", ".join(concepts[:3]),
        "context": "Interpretación local de respaldo",
        "goal": "Encontrar información pertinente en WOL",
        "concepts": concepts[:10],
        "queries": [
            " ".join(concepts[:3]),
            " ".join(concepts[:5]),
            " ".join(concepts[1:6]),
        ],
        "mode": "local",
    }

def ai_interpret(question: str):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=question,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "wol_search_interpretation",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                }
            },
            max_output_tokens=700,
        )
        import json
        data = json.loads(response.output_text)
        if len(data.get("queries", [])) != 3:
            return None
        data["mode"] = "ai"
        data["model"] = model
        return data
    except Exception as exc:
        print("AI interpretation error:", repr(exc))
        return None

@app.get("/")
def home():
    return FileResponse(BASE / "index.html")

@app.get("/health")
def health():
    return {"ok": True, "ai_configured": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/api/search")
def search(req: SearchRequest):
    result = ai_interpret(req.question)
    if result is None:
        result = local_fallback(req.question)
        result["ai_error"] = True
    result["wol_domain"] = "https://wol.jw.org/es/"
    return result
