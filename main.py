import os
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
WOL_ORIGIN = "https://wol.jw.org"
WOL_SEARCH = "https://wol.jw.org/es/wol/s/r4/lp-s"
USER_AGENT = "Mozilla/5.0 (compatible; WOLInteligente/8.0; research links)"
HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.5",
})
app = FastAPI(title="WOL Inteligente v8")


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)


PLAN_PROMPT = """
Sos el planificador de investigación de WOL Inteligente. La única fuente permitida
es la Biblioteca en línea Watchtower (WOL), en español.

No respondas la pregunta ni redactes doctrina. Tu tarea es comprender la necesidad
real del usuario y diseñar búsquedas que encuentren ARTÍCULOS concretos en WOL.

Razoná sobre:
- la pregunta explícita y lo que la persona realmente necesita resolver;
- el tema central y sus subtemas implícitos;
- expresiones, vocabulario y formas de tratar el tema que suelen aparecer en publicaciones;
- posibles términos bíblicos, nombres de funciones, situaciones prácticas y sinónimos;
- distintas perspectivas que podrían estar en artículos separados.

Generá entre 5 y 7 consultas cortas, realmente diferentes y complementarias.
No repitas la pregunta con cambios mínimos. No uses consultas genéricas como
"consejos sobre el tema". Cada consulta debe explorar un ángulo concreto.
Usá español. No incluyas nombres de sitios externos.

Devolvé JSON con queries (5 a 7) y facets (3 a 6 etiquetas breves de aspectos
que conviene cubrir). No devuelvas explicación para el usuario.
"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array", "items": {"type": "string"},
            "minItems": 5, "maxItems": 7
        },
        "facets": {
            "type": "array", "items": {"type": "string"},
            "minItems": 3, "maxItems": 6
        }
    },
    "required": ["queries", "facets"],
    "additionalProperties": False
}

EVALUATE_PROMPT = """
Sos el evaluador de pertinencia de WOL Inteligente. Los candidatos son artículos
obtenidos de páginas de WOL. La pregunta original es el criterio principal.

Leé los títulos y los fragmentos disponibles. Elegí solamente artículos cuyo
contenido aporte información directamente útil para la necesidad real, aunque
el artículo use vocabulario distinto al de la pregunta.

Reglas:
- No infieras contenido que no esté respaldado por el fragmento o el título.
- No selecciones por coincidencia superficial de palabras.
- No inventes títulos, publicaciones, enlaces ni datos.
- Priorizá artículos que traten el tema de manera directa.
- Conservá diversidad: si la pregunta tiene varios aspectos, seleccioná artículos
  distintos que cubran aspectos diferentes.
- Excluí duplicados, índices, páginas de navegación y coincidencias débiles.
- No redactes resúmenes ni respuestas.

Además, evaluá si faltan aspectos importantes de la pregunta que todavía no estén
cubiertos por los artículos candidatos. Si faltan, proponé hasta 3 búsquedas nuevas
y concretas para WOL. No repitas consultas ya realizadas.

Devolvé solo JSON:
selected: objetos con id (entero del candidato), score (1 a 5), facet (etiqueta
breve del aspecto que cubre).
missing_facets: lista de aspectos todavía no cubiertos (puede estar vacía).
followup_queries: hasta 3 consultas nuevas (puede estar vacía).
Seleccioná solo artículos con score 3, 4 o 5. Si no hay artículos pertinentes,
selected debe estar vacío.
"""

EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "facet": {"type": "string"}
                },
                "required": ["id", "score", "facet"],
                "additionalProperties": False
            },
            "maxItems": 12
        },
        "missing_facets": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6
        },
        "followup_queries": {
            "type": "array", "items": {"type": "string"}, "maxItems": 3
        }
    },
    "required": ["selected", "missing_facets", "followup_queries"],
    "additionalProperties": False
}


def call_ai(instructions: str, payload: dict, schema: dict, name: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-6")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=55, max_retries=1)
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema
                }
            },
            max_output_tokens=1600,
        )
        return json.loads(response.output_text)
    except Exception as exc:
        print(f"AI {name} error: {exc!r}")
        return None


def fallback_plan(question: str):
    q = question.lower()
    if any(x in q for x in ("atalaya", "conducir", "dirigir", "presidir")):
        return {
            "queries": [
                "conductor del Estudio de La Atalaya",
                "preparación para conducir el Estudio de La Atalaya",
                "cómo dirigir el estudio de La Atalaya participación",
                "preguntas del conductor estudio de congregación",
                "hacer comentarios participación reuniones cristianas",
                "enseñar destacar puntos principales estudio",
            ],
            "facets": ["preparación", "función del conductor", "preguntas", "participación", "enseñanza"],
        }
    if any(x in q for x in ("desanim", "triste", "abatid", "bajone")):
        return {
            "queries": [
                "cómo vencer el desánimo",
                "animar consolar a alguien desanimado",
                "fortalecer a los que están tristes",
                "palabras de ánimo consuelo",
                "ayudar espiritualmente a alguien desanimado",
            ],
            "facets": ["desánimo", "consuelo", "ánimo", "ayuda práctica"],
        }
    terms = [w.strip("¿?¡!.,;:()") for w in question.split() if len(w) > 3]
    topic = " ".join(terms[:6]) or "tema de la consulta"
    return {
        "queries": [topic, f"{topic} consejos", f"{topic} ejemplos", f"{topic} ayuda práctica", f"{topic} principios bíblicos"],
        "facets": ["tema principal", "orientación", "aplicación práctica"],
    }


def make_plan(question: str):
    plan = call_ai(PLAN_PROMPT, {"question": question}, PLAN_SCHEMA, "wol_research_plan_v8")
    if not plan or len(plan.get("queries", [])) < 5:
        plan = fallback_plan(question)
        plan["mode"] = "respaldo"
    else:
        plan["mode"] = "ia"
    # Remove empty/repeated queries while preserving order.
    plan["queries"] = list(dict.fromkeys(q.strip() for q in plan["queries"] if q.strip()))
    return plan


def search_url(query: str) -> str:
    return WOL_SEARCH + "?" + urlencode({"q": query, "r": "occ", "st": "e"})


def normalized_text(node, max_chars=3000):
    if not node:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())[:max_chars]


def is_wol_article_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == "wol.jw.org"
        and "/wol/d/" in parsed.path
    )


def fetch_search_results(query: str, limit: int = 7):
    url = search_url(query)
    try:
        response = HTTP.get(url, timeout=22)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results, seen = [], set()

        # Look for WOL document links; inspect their nearest result container
        # rather than treating every nested link as a separate result.
        for anchor in soup.select('a[href*="/wol/d/"]'):
            article_url = urljoin(WOL_ORIGIN, anchor.get("href", ""))
            title = normalized_text(anchor, 260)
            if not is_wol_article_url(article_url) or len(title) < 5:
                continue
            canonical = article_url.split("#", 1)[0]
            if canonical in seen:
                continue

            container = anchor.find_parent(["li", "article"])
            if container is None:
                for parent in anchor.parents:
                    if getattr(parent, "name", None) in ("div", "section"):
                        classes = " ".join(parent.get("class", [])).lower()
                        if any(word in classes for word in ("result", "search", "document", "item")):
                            container = parent
                            break
            context = normalized_text(container, 1100) if container else title
            results.append({
                "title": title,
                "url": canonical,
                "search_context": context,
                "search_query": query,
            })
            seen.add(canonical)
            if len(results) >= limit:
                break
        return results, None
    except Exception as exc:
        print(f"WOL search error ({query!r}): {exc!r}")
        return [], "No se pudo consultar una búsqueda de WOL."


def extract_publication(soup: BeautifulSoup) -> str:
    selectors = [
        ".publicationName", ".pubName", ".publication",
        "[class*='publicationName']", "[class*='pubName']",
        ".documentPublication"
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        value = normalized_text(node, 180)
        if value and len(value) < 180:
            return value
    return ""


def fetch_article(article: dict):
    """Lee temporalmente el texto de WOL para evaluar pertinencia; no lo almacena ni lo muestra."""
    try:
        response = HTTP.get(article["url"], timeout=22)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for node in soup.select(
            "script, style, nav, footer, header, form, "
            ".toolbar, .navigation, .publicationNavigation, "
            ".sidebar, .relatedContent, .breadcrumb"
        ):
            node.decompose()

        title_node = soup.select_one("h1, h2.documentTitle, .documentTitle")
        title = normalized_text(title_node, 260) or article["title"]
        publication = extract_publication(soup)

        content = (
            soup.select_one("#article")
            or soup.select_one("#bodyContent")
            or soup.select_one(".bodyTxt")
            or soup.select_one(".document")
            or soup.select_one("main")
            or soup.body
        )

        paragraphs = []
        if content:
            for node in content.select("p, h2, h3"):
                text = normalized_text(node, 900)
                if len(text) >= 45:
                    paragraphs.append(text)
                if sum(len(x) for x in paragraphs) >= 5200:
                    break

        excerpt = " ".join(paragraphs)[:5200]
        if not excerpt:
            excerpt = article.get("search_context", "")[:1600]

        return {
            **article,
            "title": title,
            "publication": publication,
            "content_for_evaluation_only": excerpt,
        }
    except Exception as exc:
        print(f"WOL article read error ({article['url']}): {exc!r}")
        return {
            **article,
            "publication": "",
            "content_for_evaluation_only": article.get("search_context", "")[:1600],
        }


def lexical_fallback(question: str, candidates: list):
    """Fallback conservative: scores title and WOL excerpt, never invents metadata."""
    stop = {
        "como", "para", "sobre", "cuando", "donde", "quiero", "puedo",
        "puede", "hacer", "algo", "alguien", "esto", "esta", "este",
        "desde", "tiene", "tienen", "informacion", "wol", "hay"
    }
    terms = set(re.findall(r"[a-záéíóúñü]{4,}", question.lower())) - stop
    ranked = []
    for idx, article in enumerate(candidates):
        haystack = (
            article.get("title", "") + " " +
            article.get("content_for_evaluation_only", "")[:2600]
        ).lower()
        hits = sum(1 for term in terms if term in haystack)
        if hits:
            ranked.append((hits, idx))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return {
        "selected": [
            {"id": idx, "score": min(5, max(3, hits)), "facet": "Tema relacionado"}
            for hits, idx in ranked[:8]
        ],
        "missing_facets": [],
        "followup_queries": [],
    }


def evaluate(question: str, facets: list, searched: list, candidates: list):
    payload_articles = []
    for idx, article in enumerate(candidates):
        payload_articles.append({
            "id": idx,
            "title": article["title"],
            "publication": article.get("publication", ""),
            "wol_search_context": article.get("search_context", "")[:900],
            "article_excerpt": article.get("content_for_evaluation_only", "")[:4200],
        })

    result = call_ai(
        EVALUATE_PROMPT,
        {
            "question": question,
            "research_facets": facets,
            "queries_already_used": searched,
            "articles": payload_articles,
        },
        EVALUATE_SCHEMA,
        "wol_relevance_review_v8",
    )
    return result if result is not None else lexical_fallback(question, candidates)


def format_selected(candidates: list, evaluation: dict):
    output, seen = [], set()
    selected = evaluation.get("selected", [])
    selected = sorted(selected, key=lambda item: item.get("score", 0), reverse=True)

    for item in selected:
        if item.get("score", 0) < 3:
            continue
        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
            continue
        article = candidates[idx]
        url = article["url"]
        if url in seen or not is_wol_article_url(url):
            continue
        seen.add(url)
        output.append({
            "title": article["title"],
            "url": url,
            "publication": article.get("publication", ""),
        })
    return output[:12]


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "ai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "wol_search_enabled": True,
        "version": "8.0",
    }


@app.post("/api/search")
def search(req: SearchRequest):
    plan = make_plan(req.question)
    queries = plan.get("queries", [])[:7]

    # First research pass: multiple independent searches in WOL.
    collected, seen_urls, errors = [], set(), []
    for query in queries:
        found, error = fetch_search_results(query, limit=7)
        if error:
            errors.append(error)
        for article in found:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                collected.append(article)
        time.sleep(0.12)

    # Read a bounded set of unique WOL articles. Text stays in memory and is
    # sent only to the relevance evaluator; it is never returned to the browser.
    inspected = [fetch_article(article) for article in collected[:22]]
    evaluation = evaluate(req.question, plan.get("facets", []), queries, inspected)

    # Second pass: identify coverage gaps and search only for those missing aspects.
    followups = [
        q.strip() for q in evaluation.get("followup_queries", [])
        if isinstance(q, str) and q.strip()
    ][:3]
    followups = [q for q in followups if q.lower() not in {x.lower() for x in queries}]

    if followups:
        new_candidates = []
        for query in followups:
            found, error = fetch_search_results(query, limit=6)
            if error:
                errors.append(error)
            for article in found:
                if article["url"] not in seen_urls:
                    seen_urls.add(article["url"])
                    new_candidates.append(article)
            time.sleep(0.12)

        new_inspected = [fetch_article(article) for article in new_candidates[:10]]
        if new_inspected:
            offset_candidates = inspected + new_inspected
            second_eval = evaluate(
                req.question,
                plan.get("facets", []),
                queries + followups,
                offset_candidates,
            )
            inspected = offset_candidates
            evaluation = second_eval

    articles = format_selected(inspected, evaluation)

    return {
        "version": "8.0",
        "articles": articles,
        "articles_reviewed": len(inspected),
        "searches_performed": len(queries) + len(followups),
        "message": "" if articles else "No encontramos artículos que podamos confirmar como pertinentes. Probá con otra formulación.",
        "wol_errors": list(dict.fromkeys(errors)),
        "source": "WOL",
    }
