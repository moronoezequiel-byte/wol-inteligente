import os
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()
BASE = Path(__file__).resolve().parent
WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
USER_AGENT = "Mozilla/5.0 (compatible; WOLInteligente/7.0; research links only)"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es-AR,es;q=0.9"})
app = FastAPI(title="WOL Inteligente v7")

class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)

INTERPRET_PROMPT = """Sos el motor de comprensión de WOL Inteligente. Tu única tarea es convertir la pregunta en un plan de investigación dentro de la Biblioteca en línea Watchtower (WOL), en español.
NO respondas la pregunta. NO redactes doctrina. NO uses fuentes externas.
Identificá la necesidad real, el tema, el contexto y las distintas subnecesidades que conviene investigar. Pensá en sinónimos, expresiones habituales de publicaciones y conceptos relacionados que puedan aparecer aunque el usuario no los haya escrito.
Generá entre 4 y 6 consultas cortas, variadas y complementarias para buscar en WOL. Evitá consultas repetidas o que solo reordenen las mismas palabras. Incluí consultas sobre las subnecesidades implícitas. Usá frases de búsqueda naturales, no preguntas largas.
Devolvé JSON con intent, topic, context, needs (lista de 3 a 6 necesidades breves) y queries (lista de 4 a 6 consultas)."""

INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "topic": {"type": "string"},
        "context": {"type": "string"},
        "needs": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6},
        "queries": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 6}
    },
    "required": ["intent", "topic", "context", "needs", "queries"],
    "additionalProperties": False
}

RANK_PROMPT = """Sos un evaluador de pertinencia para un buscador de Watchtower ONLINE LIBRARY (WOL).
Vas a recibir la pregunta original y una lista de artículos que fueron obtenidos directamente de WOL. Para cada artículo tenés título, publicación si se pudo detectar y un fragmento de su propio contenido.
Evaluá si el artículo contiene información útil para la necesidad expresada, incluso si usa otras palabras. No agregues conocimiento externo, no completes lo que el fragmento no dice y no inventes datos.
No escribas resúmenes ni respondas la pregunta. Solo seleccioná los artículos realmente pertinentes y asignales una categoría breve basada en la necesidad que cubren. Preferí diversidad de aspectos y evitá duplicados. Excluí coincidencias superficiales.
Devolvé únicamente JSON con selected: lista de objetos {id, relevance, need}. relevance es un entero de 1 a 5; need es una etiqueta breve (por ejemplo: preparación, participación, preguntas, enseñanza). No incluyas texto explicativo."""

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "need": {"type": "string"}
                },
                "required": ["id", "relevance", "need"],
                "additionalProperties": False
            },
            "maxItems": 12
        }
    },
    "required": ["selected"],
    "additionalProperties": False
}


def openai_json(instructions: str, payload: dict, schema: dict, schema_name: str):
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-6")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=45, max_retries=1)
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
            max_output_tokens=1200,
        )
        return json.loads(response.output_text)
    except Exception as exc:
        print(f"OpenAI {schema_name} error: {exc!r}")
        return None


def fallback_plan(question: str):
    q = question.lower()
    if any(w in q for w in ["atalaya", "conducir", "dirigir", "presidir"]):
        topic = "Conducción del Estudio de La Atalaya"
        needs = ["preparación del conductor", "formulación de preguntas", "participación del auditorio", "enseñanza y puntos principales"]
        queries = ["conductor del Estudio de La Atalaya", "preparación para conducir el estudio", "preguntas y participación Estudio de La Atalaya", "dirigir estudio congregación enseñanza"]
    elif any(w in q for w in ["desanim", "triste", "abatid"]):
        topic = "Desánimo y cómo animar"
        needs = ["causas del desánimo", "consolar y animar", "fortalecer a alguien"]
        queries = ["desánimo", "animar consolar fortalecer", "ayudar a alguien desanimado", "palabras de ánimo consuelo"]
    else:
        terms = [x.strip("¿?¡!.,;:") for x in question.split() if len(x) > 3]
        topic = " ".join(terms[:6]) or "Información relacionada con la consulta"
        needs = ["tema principal", "consejos relacionados", "situaciones prácticas"]
        queries = [topic, "consejos " + topic, "ayuda " + topic, "orientación práctica " + topic]
    return {"intent": "Encontrar artículos de WOL que respondan a la necesidad", "topic": topic, "context": "Plan de búsqueda local de respaldo", "needs": needs, "queries": queries, "mode": "local"}


def interpret(question: str):
    data = openai_json(INTERPRET_PROMPT, {"question": question}, INTERPRET_SCHEMA, "wol_research_plan")
    if not data:
        return fallback_plan(question)
    data["mode"] = "ai"
    return data


def search_url(query: str) -> str:
    return WOL_SEARCH + "?" + urlencode({"q": query, "r": "occ", "st": "e"})


def clean_text(node, max_chars=2200):
    if not node:
        return ""
    text = " ".join(node.get_text(" ", strip=True).split())
    return text[:max_chars]


def fetch_search_results(query: str, limit=10):
    url = search_url(query)
    try:
        response = HTTP.get(url, timeout=18)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        found, seen = [], set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            full_url = urljoin(WOL_ORIGIN, href)
            title = clean_text(a, 240)
            if "/wol/d/" not in full_url or not title or len(title) < 5:
                continue
            # WOL article documents are under /wol/d/; exclude links outside official host.
            if not full_url.startswith(WOL_ORIGIN + "/") or full_url in seen:
                continue
            seen.add(full_url)
            parent = a.find_parent(["li", "article", "div"])
            context = clean_text(parent, 900) if parent else title
            found.append({"title": title, "url": full_url, "search_query": query, "search_context": context})
            if len(found) >= limit:
                break
        return found, None
    except Exception as exc:
        print(f"WOL search error for {query!r}: {exc!r}")
        return [], "No se pudo consultar esta búsqueda en WOL."


def fetch_article(article: dict):
    """Read a limited excerpt in memory only to assess relevance; never display or store article text."""
    try:
        response = HTTP.get(article["url"], timeout=18)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for unwanted in soup.select("script, style, nav, footer, header, .toolbar, .navigation, .publicationNavigation"):
            unwanted.decompose()
        title_node = soup.find(["h1", "h2"])
        title = clean_text(title_node, 300) or article["title"]
        # WOL pages commonly mark article text in bodyContent; fall back to main/body.
        content = soup.select_one("#article, #bodyContent, .bodyTxt, main") or soup.body
        paragraphs = []
        if content:
            for p in content.find_all(["p", "h2", "h3"]):
                t = clean_text(p, 700)
                if len(t) > 35:
                    paragraphs.append(t)
                if sum(map(len, paragraphs)) >= 4500:
                    break
        excerpt = " ".join(paragraphs)[:4500]
        publication = ""
        meta = soup.select_one(".publicationName, .pubName, [class*=publication]")
        if meta:
            publication = clean_text(meta, 180)
        if not excerpt:
            excerpt = article.get("search_context", "")
        return {**article, "title": title, "publication": publication, "article_text_for_relevance_only": excerpt}
    except Exception as exc:
        print(f"WOL article fetch error {article['url']}: {exc!r}")
        return {**article, "publication": "", "article_text_for_relevance_only": article.get("search_context", "")}


def rank_articles(question: str, plan: dict, articles: list):
    if not articles:
        return []
    payload_articles = []
    for i, a in enumerate(articles):
        payload_articles.append({"id": i, "title": a["title"], "publication": a.get("publication", ""), "content_excerpt": a.get("article_text_for_relevance_only", "")[:3500]})
    ranked = openai_json(
        RANK_PROMPT,
        {"question": question, "research_needs": plan.get("needs", []), "articles": payload_articles},
        RANK_SCHEMA,
        "wol_article_relevance",
    )
    if not ranked:
        # Conservative fallback: use title/context matching only, with no generated claims.
        words = [w for w in re.findall(r"[a-záéíóúñü]{4,}", question.lower())]
        ranked_items = []
        for i, a in enumerate(articles):
            haystack = (a["title"] + " " + a.get("search_context", "")).lower()
            score = sum(1 for w in set(words) if w in haystack)
            if score:
                ranked_items.append((score, i))
        ranked_items.sort(reverse=True)
        selected = [{"id": i, "relevance": min(5, score), "need": "Coincidencia temática"} for score, i in ranked_items[:8]]
    else:
        selected = ranked.get("selected", [])
    output, used = [], set()
    for item in sorted(selected, key=lambda x: x.get("relevance", 0), reverse=True):
        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= len(articles) or idx in used:
            continue
        used.add(idx)
        a = articles[idx]
        output.append({"title": a["title"], "url": a["url"], "publication": a.get("publication", ""), "need": item.get("need", "Relacionado"), "relevance": item.get("relevance", 0)})
    return output[:12]


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")

@app.get("/health")
def health():
    return {"ok": True, "ai_configured": bool(os.getenv("OPENAI_API_KEY")), "wol_search_enabled": True, "version": "7.0"}

@app.post("/api/search")
def search(req: SearchRequest):
    plan = interpret(req.question)
    all_results, errors, searched = [], [], []
    seen = set()
    for query in plan.get("queries", [])[:6]:
        results, error = fetch_search_results(query, limit=8)
        searched.append(query)
        if error:
            errors.append(error)
        for item in results:
            if item["url"] not in seen:
                seen.add(item["url"])
                all_results.append(item)
        time.sleep(0.15)
    # Read article content to assess relevance, but do not send the article text to the UI.
    inspected = [fetch_article(a) for a in all_results[:24]]
    articles = rank_articles(req.question, plan, inspected)
    return {
        "version": "7.0",
        "mode": plan.get("mode", "local"),
        "interpretation": {k: plan.get(k) for k in ["intent", "topic", "context", "needs"]},
        "searched_queries": searched,
        "articles": articles,
        "articles_inspected": len(inspected),
        "message": "" if articles else ("WOL no devolvió artículos que podamos confirmar como pertinentes. Probá reformular la consulta."),
        "wol_errors": list(dict.fromkeys(errors)),
        "source": "WOL"
    }
