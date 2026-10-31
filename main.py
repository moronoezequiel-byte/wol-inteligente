import os,json,re,time,unicodedata
from pathlib import Path
from urllib.parse import urlencode,urljoin,urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field

load_dotenv()
BASE=Path(__file__).resolve().parent
ORIGIN="https://wol.jw.org"
SEARCH="https://wol.jw.org/es/wol/s/r4/lp-s"
HTTP=requests.Session()
HTTP.headers.update({"User-Agent":"Mozilla/5.0 (compatible; WOLInteligente/11.0)","Accept-Language":"es-AR,es;q=0.9"})
app=FastAPI(title="WOL Inteligente v11")

class SearchRequest(BaseModel):
    question:str=Field(min_length=3,max_length=1500)

PLAN_PROMPT="""Sos el investigador principal de WOL Inteligente. La unica fuente es WOL en espanol. NO respondas la pregunta.
Tu trabajo es modelar exactamente que necesita encontrar el usuario y preparar rutas de investigacion.
Separá intencion, accion, objeto, contexto, rol, problema, resultado, condiciones y conceptos imprescindibles.
No supongas La Atalaya salvo que la pregunta lo indique. Genera consultas distintas, concretas y plausibles para WOL.
No inventes terminologia como si fuera WOL: marcala como hipotesis y deja que la investigacion la confirme.
Devuelve JSON interno."""
PLAN_SCHEMA={"type":"object","properties":{
"intent":{"type":"string"},"action":{"type":"string"},"object":{"type":"string"},"context":{"type":"string"},"role":{"type":"string"},"goal":{"type":"string"},
"concepts":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":12},
"must_match":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":10},
"avoid":{"type":"array","items":{"type":"string"},"maxItems":10},
"queries":{"type":"array","items":{"type":"string"},"minItems":8,"maxItems":16}},
"required":["intent","action","object","context","role","goal","concepts","must_match","avoid","queries"],"additionalProperties":False}

VOCAB_PROMPT="""Sos un investigador que acaba de leer resultados REALES de WOL. No respondas la pregunta.
Analiza los titulos y fragmentos proporcionados y extrae solamente vocabulario que WOL realmente usa y que pueda ayudar a investigar la necesidad.
Distingue terminos centrales de terminos accidentales. Descarta palabras genericas. Identifica expresiones, nombres de funciones, acciones, problemas, contextos y relaciones que aparezcan realmente en los documentos.
Devuelve nuevas consultas basadas en vocabulario descubierto, no una lista de sinonimos inventados."""
VOCAB_SCHEMA={"type":"object","properties":{
"confirmed_terms":{"type":"array","items":{"type":"string"},"maxItems":20},
"useful_phrases":{"type":"array","items":{"type":"string"},"maxItems":12},
"new_queries":{"type":"array","items":{"type":"string"},"maxItems":12},
"gaps":{"type":"array","items":{"type":"string"},"maxItems":8}},
"required":["confirmed_terms","useful_phrases","new_queries","gaps"],"additionalProperties":False}

JUDGE_PROMPT="""Sos el juez de pertinencia de una investigacion en WOL. Los candidatos son articulos REALES.
La pregunta completa y su estructura son el criterio. Un articulo SOLO es pertinente si aporta evidencia sobre la necesidad concreta del usuario.
Una coincidencia de una palabra, tema general, mencion pasajera, contexto diferente o relacion tangencial NO alcanza.
Si el usuario pregunta COMO hacer algo, un articulo debe aportar orientacion sobre esa accion, responsabilidad, procedimiento, habilidad o condiciones, no simplemente mencionar el tema.
Exige correspondencia entre accion+objeto+contexto cuando esos elementos existan. Para preguntas doctrinales exige que el articulo trate realmente esa cuestion.
Para cada candidato seleccionado indica evidencia concreta del texto, que parte de la necesidad cubre y posibles dudas. Selecciona solo score 4 o 5. Si ninguno cumple, devuelve lista vacia.
Genera followups solo para huecos reales y no repitas consultas ya usadas."""
JUDGE_SCHEMA={"type":"object","properties":{
"selected":{"type":"array","items":{"type":"object","properties":{
"id":{"type":"integer"},"score":{"type":"integer","minimum":4,"maximum":5},"coverage":{"type":"string"},"evidence":{"type":"string"},"doubt":{"type":"string"}},
"required":["id","score","coverage","evidence","doubt"],"additionalProperties":False},"maxItems":20},
"missing":{"type":"array","items":{"type":"string"},"maxItems":10},
"followups":{"type":"array","items":{"type":"string"},"maxItems":8}},"required":["selected","missing","followups"],"additionalProperties":False}

AUDIT_PROMPT="""Sos un auditor adversarial independiente. Tu objetivo es intentar DESCARTAR cada articulo seleccionado.
No confies en la evaluacion anterior. Lee la pregunta, la estructura y el articulo. Un resultado sobrevive solo si la evidencia demuestra pertinencia directa.
Penaliza coincidencias superficiales, palabras compartidas, menciones pasajeras, contexto distinto, articulo demasiado amplio y articulos que solo rozan el asunto.
Conserva solo los que superen una prueba estricta. No inventes informacion."""
AUDIT_SCHEMA={"type":"object","properties":{
"survivors":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"reason":{"type":"string"}},"required":["id","reason"],"additionalProperties":False},"maxItems":16},
"rejected":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"reason":{"type":"string"}},"required":["id","reason"],"additionalProperties":False},"maxItems":20},
"need_more_research":{"type":"boolean"},"gaps":{"type":"array","items":{"type":"string"},"maxItems":8}},
"required":["survivors","rejected","need_more_research","gaps"],"additionalProperties":False}

FINAL_PROMPT="""Sos el auditor final de una investigacion WOL. Decide el conjunto MINIMO de articulos que realmente satisface la necesidad.
Solo puedes elegir articulos que sobrevivieron la auditoria y tienen evidencia. Elimina redundancias y resultados genericos.
No rellenes cantidad. Es valido devolver 0, 1, 2 o varios. Cada articulo debe cubrir una parte concreta de la pregunta.
Devuelve solo IDs finales."""
FINAL_SCHEMA={"type":"object","properties":{"final_ids":{"type":"array","items":{"type":"integer"},"maxItems":12}},"required":["final_ids"],"additionalProperties":False}

def ai(instructions,payload,schema,name,tokens=3000):
    key=os.getenv("OPENAI_API_KEY")
    if not key:return None
    try:
        from openai import OpenAI
        r=OpenAI(api_key=key,timeout=75,max_retries=1).responses.create(
            model=os.getenv("OPENAI_MODEL","gpt-6"),instructions=instructions,
            input=json.dumps(payload,ensure_ascii=False),
            text={"format":{"type":"json_schema","name":name,"strict":True,"schema":schema}},
            max_output_tokens=tokens)
        return json.loads(r.output_text)
    except Exception as e:
        print(f"AI {name} error: {e!r}");return None

def fold(s):
    s=unicodedata.normalize("NFD",str(s).lower())
    return "".join(c for c in s if unicodedata.category(c)!="Mn")

STOP={"como","para","sobre","cuando","donde","quiero","puedo","puede","hacer","algo","alguien","esto","esta","este","desde","tiene","tienen","informacion","buscar","encontrar","articulos","biblioteca","linea","wol","dice","decir","hay"}
def toks(s): return {x for x in re.findall(r"[a-záéíóúñü]{3,}",fold(s)) if x not in STOP}

def fallback(q):
    t=list(toks(q))[:8]
    qs=[q]
    if len(t)>=2:
        qs += [" ".join(t[:4])," ".join(t[:5])," ".join(t[:3])+" consejos"," ".join(t[:4])+" responsabilidades"," ".join(t[:4])+" preparación"," ".join(t[:4])+" ayudar"," ".join(t[:4])+" dificultades"]
    return {"intent":q,"action":"","object":"","context":"","role":"","goal":"","concepts":t,"must_match":t[:5],"avoid":[],"queries":qs}

def plan(q):
    p=ai(PLAN_PROMPT,{"question":q},PLAN_SCHEMA,"wol_research_map_v11",3000)
    if not p:p=fallback(q)
    out=[];seen=set()
    for x in p.get("queries",[]):
        if isinstance(x,str) and x.strip() and fold(x) not in seen:
            seen.add(fold(x));out.append(x.strip())
    p["queries"]=out[:16]
    return p

def search_url(q):return SEARCH+"?"+urlencode({"q":q,"r":"occ","st":"e"})
def text(n,nmax=3000):return " ".join(n.get_text(" ",strip=True).split())[:nmax] if n else ""
def article_url(u):
    p=urlparse(u);return p.scheme=="https" and p.netloc.lower()=="wol.jw.org" and "/wol/d/" in p.path

def wol_search(q,limit=10):
    try:
        r=HTTP.get(search_url(q),timeout=25,allow_redirects=True);r.raise_for_status()
        s=BeautifulSoup(r.text,"html.parser");out=[];seen=set()
        for a in s.find_all("a",href=True):
            h=a.get("href","").strip()
            if "/wol/d/" not in h:continue
            u=urljoin(ORIGIN,h).split("#")[0]
            if not article_url(u) or u in seen:continue
            title=text(a.find(["h1","h2","h3","h4"]) or a,360)
            if len(title)<5:continue
            box=a.find_parent(["li","article"])
            if not box:
                for p in a.parents:
                    if getattr(p,"name",None) not in ("div","section"):continue
                    mark=(" ".join(p.get("class",[]))+" "+" ".join(map(str,p.attrs.values()))).lower()
                    if any(x in mark for x in ("result","search","document","entry","item")):
                        box=p;break
            out.append({"title":title,"url":u,"context":text(box,1800) if box else title,"query":q});seen.add(u)
            if len(out)>=limit:break
        print(f"WOL q={q!r} status={r.status_code} results={len(out)}")
        return out,None
    except Exception as e:
        print(f"WOL search error {q!r}: {e!r}");return [],f"No se pudo consultar una busqueda de WOL: {e}"

def publication(s):
    for sel in [".publicationName",".pubName",".publication","[class*='publicationName']","[class*='pubName']",".documentPublication"]:
        v=text(s.select_one(sel),180)
        if v:return v
    return ""

def read_article(a):
    try:
        r=HTTP.get(a["url"],timeout=22);r.raise_for_status();s=BeautifulSoup(r.text,"html.parser")
        for n in s.select("script,style,nav,footer,header,form,.toolbar,.navigation,.publicationNavigation,.sidebar,.relatedContent,.breadcrumb"):n.decompose()
        title=text(s.select_one("h1,h2.documentTitle,.documentTitle"),300) or a["title"]
        body=s.select_one("#article") or s.select_one("#bodyContent") or s.select_one(".bodyTxt") or s.select_one(".document") or s.select_one("main") or s.body
        parts=[]
        if body:
            for n in body.select("p,h2,h3,li"):
                v=text(n,1200)
                if len(v)>=35:parts.append(v)
                if sum(map(len,parts))>=7500:break
        return {**a,"title":title,"publication":publication(s),"excerpt":" ".join(parts)[:7500] or a["context"][:1800]}
    except Exception as e:
        print(f"WOL article error {a['url']}: {e!r}")
        return {**a,"publication":"","excerpt":a["context"][:1800]}

def discovery_payload(arts):
    return [{"id":i,"title":a["title"],"found_by":a.get("query",""),"context":a.get("context","")[:1400],"excerpt":a.get("excerpt","")[:5000]} for i,a in enumerate(arts)]

def discover_vocab(question,p,arts):
    return ai(VOCAB_PROMPT,{"question":question,"research_map":p,"documents":discovery_payload(arts)},VOCAB_SCHEMA,"wol_vocab_discovery_v11",2800)

def judge(question,p,used,arts):
    return ai(JUDGE_PROMPT,{"question":question,"research_map":p,"queries_used":used,"articles":discovery_payload(arts)},JUDGE_SCHEMA,"wol_relevance_judge_v11",4200)

def audit(question,p,arts,selected):
    ids={x.get("id") for x in selected if isinstance(x,dict)}
    chosen=[{"id":i,"title":a["title"],"publication":a.get("publication",""),"excerpt":a.get("excerpt","")[:7000]} for i,a in enumerate(arts) if i in ids]
    return ai(AUDIT_PROMPT,{"question":question,"research_map":p,"selected_articles":chosen},AUDIT_SCHEMA,"wol_adversarial_audit_v11",3500)

def pre_rank(p,a):
    terms=set()
    for k in ("concepts","must_match"):
        for x in p.get(k,[]):terms|=toks(x)
    title=toks(a.get("title",""));ctx=toks(a.get("context",""))
    return 8*len(terms&title)+2*len(terms&ctx)

def merge_unique(base,extra):
    seen={a["url"] for a in base}
    for a in extra:
        if a["url"] not in seen:seen.add(a["url"]);base.append(a)

def do_searches(queries,seen,errors):
    out=[]
    for q in queries:
        found,err=wol_search(q)
        if err:errors.append(err)
        for a in found:
            if a["url"] not in seen:seen.add(a["url"]);out.append(a)
        time.sleep(.08)
    return out

def final_results(arts,audit_result,final_result):
    survivors={x.get("id") for x in (audit_result or {}).get("survivors",[]) if isinstance(x,dict)}
    ids=[x for x in (final_result or {}).get("final_ids",[]) if isinstance(x,int) and x in survivors]
    out=[];seen=set()
    for i in ids:
        if i<0 or i>=len(arts):continue
        a=arts[i]
        if a["url"] in seen:continue
        seen.add(a["url"]);out.append({"title":a["title"],"url":a["url"],"publication":a.get("publication","")})
    return out[:12]

def research(q):
    p=plan(q);errors=[];seen=set();queries=[]
    first=p.get("queries",[])[:16];queries.extend(first)
    candidates=do_searches(first,seen,errors)
    candidates.sort(key=lambda a:pre_rank(p,a),reverse=True)
    inspected=[read_article(a) for a in candidates[:45]]

    vocab=discover_vocab(q,p,inspected) if inspected else None
    vocab_queries=[] if not vocab else [x.strip() for x in vocab.get("new_queries",[]) if isinstance(x,str) and x.strip()]
    used={fold(x) for x in queries}
    vocab_queries=[x for x in vocab_queries if fold(x) not in used][:12]
    if vocab_queries:
        queries.extend(vocab_queries)
        extra=do_searches(vocab_queries,seen,errors)
        extra.sort(key=lambda a:pre_rank(p,a),reverse=True)
        merge_unique(inspected,[read_article(a) for a in extra[:30]])

    judge_result=judge(q,p,queries,inspected) if inspected else None
    selected=(judge_result or {}).get("selected",[]) if judge_result else []

    follow=[] if not judge_result else judge_result.get("followups",[])
    used={fold(x) for x in queries}
    follow=[x.strip() for x in follow if isinstance(x,str) and x.strip() and fold(x) not in used][:8]
    if follow:
        queries.extend(follow)
        extra=do_searches(follow,seen,errors)
        extra.sort(key=lambda a:pre_rank(p,a),reverse=True)
        merge_unique(inspected,[read_article(a) for a in extra[:28]])
        judge_result=judge(q,p,queries,inspected) if inspected else judge_result
        selected=(judge_result or {}).get("selected",[]) if judge_result else []

    audit_result=audit(q,p,inspected,selected) if selected else {"survivors":[],"rejected":[],"need_more_research":False,"gaps":[]}

    if audit_result.get("need_more_research") and len(queries)<32:
        gapq=[x.strip() for x in audit_result.get("gaps",[]) if isinstance(x,str) and x.strip() and fold(x) not in {fold(y) for y in queries}][:6]
        if gapq:
            queries.extend(gapq)
            extra=do_searches(gapq,seen,errors)
            extra.sort(key=lambda a:pre_rank(p,a),reverse=True)
            merge_unique(inspected,[read_article(a) for a in extra[:24]])
            judge_result=judge(q,p,queries,inspected) if inspected else judge_result
            selected=(judge_result or {}).get("selected",[]) if judge_result else []
            audit_result=audit(q,p,inspected,selected) if selected else {"survivors":[],"rejected":[],"need_more_research":False,"gaps":[]}

    survivors=audit_result.get("survivors",[]) if audit_result else []
    final_result=ai(FINAL_PROMPT,{"question":q,"research_map":p,"survivors":survivors,"articles":discovery_payload(inspected)},FINAL_SCHEMA,"wol_final_selection_v11",2200) if survivors else {"final_ids":[]}
    results=final_results(inspected,audit_result,final_result)

    return {"version":"11.0","articles":results,"articles_reviewed":len(inspected),"searches_performed":len(queries),"candidate_articles_found":len(seen),
            "message":"" if results else "No encontramos artículos que hayan superado la investigación y auditoría de pertinencia. Probá con otra formulación.",
            "wol_errors":list(dict.fromkeys(errors)),"source":"WOL"}

@app.get("/")
def home():return FileResponse(BASE/"index.html")
@app.get("/health")
def health():return {"ok":True,"ai_configured":bool(os.getenv("OPENAI_API_KEY")),"wol_search_enabled":True,"version":"11.0"}
@app.post("/api/search")
def search(req:SearchRequest):return research(req.question)
