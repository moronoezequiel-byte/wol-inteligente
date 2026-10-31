import os
import json
import re
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
WOL_BASE = "https://wol.jw.org/es/"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
app = FastAPI(title="WOL Inteligente v6")

class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)

SYSTEM_PROMPT = """
Sos el intérprete semántico de WOL Inteligente.

Tu única función es transformar una pregunta natural en una estrategia de búsqueda
para la Biblioteca en línea Watchtower (WOL). NO respondas la pregunta. NO redactes
doctrina. NO uses fuentes externas. NO inventes contenido.

Interpretá intención, tema, contexto, objetivo y conceptos útiles para encontrar
información en WOL. Generá exactamente 3 consultas concisas y distintas:
1) tema principal;
2) sinónimos o términos relacionados;
3) situación/objetivo.

La salida debe contener: intent, topic, context, goal, concepts y queries.
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
        data = json.loads(response.output_text)
        if len(data.get("queries", [])) != 3:
            return None
        data["mode"] = "ai"
        data["model"] = model
        return data
    except Exception as exc:
        print("AI interpretation error:", repr(exc))
        return None

def wol_search_url(query: str) -> str:
    return f"{WOL_SEARCH}?q={quote_plus(query)}&r=occ&st=e"

def fetch_wol_results(query: str, limit: int = 5):
    """Obtiene títulos/enlaces de la página de resultados de WOL.
    No guarda el contenido: solo usa la respuesta para mostrar resultados al usuario.
    """
    url = wol_search_url(query)
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "WOL-Inteligente/6.0 (búsqueda de resultados WOL)"
            },
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        results = []
        seen = set()

        # WOL suele presentar resultados como enlaces a /wol/d/...
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            title = " ".join(a.get_text(" ", strip=True).split())

            if not title:
                continue

            absolute = urljoin("https://wol.jw.org", href)
            if "/wol/d/" not in absolute:
                continue
            if absolute in seen:
                continue

            # Evita enlaces de navegación que no sean artículos.
            if len(title) < 4:
                continue

            seen.add(absolute)
            results.append({
                "title": title,
                "url": absolute,
                "query": query,
            })

            if len(results) >= limit:
                break

        return {
            "query": query,
            "search_url": url,
            "results": results,
            "error": None,
        }

    except Exception as exc:
        print("WOL search error:", repr(exc))
        return {
            "query": query,
            "search_url": url,
            "results": [],
            "error": "No se pudo consultar WOL en este momento.",
        }

@app.get("/")
def home():
    return FileResponse(BASE / "index.html")

@app.get("/health")
def health():
    return {
        "ok": True,
        "ai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "wol_enabled": True,
    }

@app.post("/api/search")
def search(req: SearchRequest):
    result = ai_interpret(req.question)
    if result is None:
        result = local_fallback(req.question)
        result["ai_error"] = True

    queries = result.get("queries", [])
    wol_results = [fetch_wol_results(q, limit=5) for q in queries]

    # Deduplicar artículos manteniendo el mejor orden de aparición.
    merged = []
    seen = set()
    for group in wol_results:
        for item in group["results"]:
            if item["url"] not in seen:
                seen.add(item["url"])
                merged.append(item)

    result["wol_domain"] = WOL_BASE
    result["wol_searches"] = wol_results
    result["articles"] = merged[:12]
    return result
