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
HTTP.headers.update({"User-Agent":"Mozilla/5.0 (compatible; WOLInteligente/10.0)","Accept-Language":"es-AR,es;q=0.9"})
app=FastAPI(title="WOL Inteligente v10")

class SearchRequest(BaseModel):
    question:str=Field(min_length=3,max_length=1500)

PLAN_PROMPT="""Sos el estratega de investigación de WOL Inteligente. La única fuente es la Biblioteca en línea Watchtower (WOL), en español.
NO contestes la pregunta. Descubrí qué información necesita realmente el usuario y cómo podría aparecer redactada en publicaciones.
Pensá como investigador: intención, conceptos, situación/problema, acción o habilidad, función/rol, sinónimos y vocabulario típico de publicaciones.
No asumas que el tema es La Atalaya. Debe funcionar para cualquier pregunta.
Generá búsquedas realmente distintas: directa, sinónimos, problema subyacente, acción/habilidad, contexto/rol, vocabulario probable de WOL y una búsqueda amplia de descubrimiento.
Devolvé JSON con intent, concepts, queries (8-12), must_match y avoid. Es información interna y nunca se muestra al usuario."""
PLAN_SCHEMA={"type":"object","properties":{
"intent":{"type":"string"},
"concepts":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":8},
"queries":{"type":"array","items":{"type":"string"},"minItems":8,"maxItems":12},
"must_match":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":8},
"avoid":{"type":"array","items":{"type":"string"},"maxItems":6}},
"required":["intent","concepts","queries","must_match","avoid"],"additionalProperties":False}

EVAL_PROMPT="""Sos un evaluador estricto de pertinencia para WOL Inteligente.
La pregunta y el mapa de investigación son el criterio. Los candidatos son artículos reales de WOL.
Seleccioná SOLO artículos que probablemente aporten información directamente útil para la necesidad real.
Una coincidencia de una palabra NO alcanza. Si el artículo solo menciona el tema de pasada, descartalo.
Priorizá artículos directos, diversos y específicos. Si el usuario pregunta cómo hacer algo, buscá material que explique cómo hacerlo.
No inventes información ni redactes respuestas.
Asigná score 5=directo y muy útil, 4=claramente pertinente, 3=pertinente pero indirecto.
Devolvé selected, missing y followups. Las followups deben cubrir huecos reales y ser búsquedas nuevas."""
EVAL_SCHEMA={"type":"object","properties":{
"selected":{"type":"array","items":{"type":"object","properties":{
"id":{"type":"integer"},"score":{"type":"integer","minimum":1,"maximum":5},"tag":{"type":"string"}},
"required":["id","score","tag"],"additionalProperties":False},"maxItems":16},
"missing":{"type":"array","items":{"type":"string"},"maxItems":6},
"followups":{"type":"array","items":{"type":"string"},"maxItems":4}},
"required":["selected","missing","followups"],"additionalProperties":False}

def ai(instructions,payload,schema,name,tokens=2400):
    key=os.getenv("OPENAI_API_KEY")
    if not key:return None
    try:
        from openai import OpenAI
        r=OpenAI(api_key=key,timeout=60,max_retries=1).responses.create(
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

STOP={"como","para","sobre","cuando","donde","quiero","puedo","puede","hacer","algo","alguien","esto","esta","este","desde","tiene","tienen","informacion","buscar","encontrar","articulos","biblioteca","linea"}
def toks(s): return {x for x in re.findall(r"[a-záéíóúñü]{3,}",fold(s)) if x not in STOP}

def fallback(q):
    t=list(toks(q))[:7]; topic=" ".join(t) or q
    qs=[q,topic,f"{topic} cómo",f"{topic} consejos",f"{topic} ejemplos",
        f"{topic} ayuda práctica",f"{topic} experiencia",f"{topic} principios bíblicos",
        f"{topic} problemas dificultades",f"{topic} responsabilidades función"]
    return {"intent":q,"concepts":t,"queries":qs,"must_match":t[:5],"avoid":[]}

def plan(q):
    p=ai(PLAN_PROMPT,{"question":q},PLAN_SCHEMA,"wol_research_map_v10")
    if not p or len(p.get("queries",[]))<8:p=fallback(q)
    out=[];seen=set()
    for x in p.get("queries",[]):
        if isinstance(x,str) and x.strip() and fold(x) not in seen:
            seen.add(fold(x));out.append(x.strip())
    p["queries"]=out[:12];return p

def search_url(q):return SEARCH+"?"+urlencode({"q":q,"r":"occ","st":"e"})
def text(n,nmax=3000):return " ".join(n.get_text(" ",strip=True).split())[:nmax] if n else ""
def article_url(u):
    p=urlparse(u);return p.scheme=="https" and p.netloc.lower()=="wol.jw.org" and "/wol/d/" in p.path

def wol_search(q,limit=9):
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
            out.append({"title":title,"url":u,"context":text(box,1500) if box else title,"query":q});seen.add(u)
            if len(out)>=limit:break
        print(f"WOL q={q!r} status={r.status_code} results={len(out)}")
        return out,None
    except Exception as e:
        print(f"WOL search error {q!r}: {e!r}");return [],"No se pudo consultar una búsqueda de WOL."

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
                v=text(n,1000)
                if len(v)>=35:parts.append(v)
                if sum(map(len,parts))>=6500:break
        return {**a,"title":title,"publication":publication(s),"excerpt":" ".join(parts)[:6500] or a["context"][:1800]}
    except Exception as e:
        print(f"WOL article error {a['url']}: {e!r}")
        return {**a,"publication":"","excerpt":a["context"][:1800]}

def pre_score(q,p,a):
    allterms=toks(q)
    for c in p.get("concepts",[]):allterms|=toks(c)
    must=set()
    for c in p.get("must_match",[]):must|=toks(c)
    title=fold(a["title"]);hay=title+" "+fold(a["context"])
    return sum(t in title for t in allterms)*7+sum(t in hay for t in allterms)*2+sum(t in hay for t in must)*3

def local(q,p,cands):
    qterms=toks(q);ct=set()
    for c in p.get("concepts",[]):ct|=toks(c)
    mt=set()
    for c in p.get("must_match",[]):mt|=toks(c)
    rank=[]
    for i,a in enumerate(cands):
        hay=fold(a["title"]+" "+a.get("excerpt",""))
        score=sum(t in hay for t in qterms)+sum(t in hay for t in ct)+2*sum(t in hay for t in mt)+4*sum(t in fold(a["title"]) for t in qterms|ct|mt)
        if score>=3:rank.append((score,i))
    rank.sort(reverse=True)
    return {"selected":[{"id":i,"score":3,"tag":"Coincidencia temática"} for _,i in rank[:8]],"missing":[],"followups":[]}

def evaluate(q,p,used,cands):
    arts=[{"id":i,"title":a["title"],"publication":a.get("publication",""),"found_by":a.get("query",""),
           "search_context":a.get("context","")[:1200],"article_excerpt":a.get("excerpt","")[:5200]}
          for i,a in enumerate(cands)]
    r=ai(EVAL_PROMPT,{"question":q,"research_map":{"intent":p.get("intent",""),"concepts":p.get("concepts",[]),
       "must_match":p.get("must_match",[]),"avoid":p.get("avoid",[])},
       "queries_already_used":used,"articles":arts},EVAL_SCHEMA,"wol_relevance_review_v10",2800)
    if r is None:return local(q,p,cands)
    if not r.get("selected"):
        l=local(q,p,cands)
        if l["selected"]:r["selected"]=l["selected"][:6]
    return r

def format_results(cands,e):
    out=[];seen=set()
    for x in sorted(e.get("selected",[]),key=lambda x:-x.get("score",0)):
        i=x.get("id");score=x.get("score",0)
        if score<3 or not isinstance(i,int) or i<0 or i>=len(cands):continue
        a=cands[i]
        if a["url"] in seen:continue
        seen.add(a["url"]);out.append({"title":a["title"],"url":a["url"],"publication":a.get("publication","")})
    return out[:12]

def research(q):
    p=plan(q);queries=p["queries"][:12];allc=[];seen=set();errors=[]
    for qq in queries:
        found,err=wol_search(qq)
        if err:errors.append(err)
        for a in found:
            if a["url"] not in seen:seen.add(a["url"]);allc.append(a)
        time.sleep(.10)
    allc.sort(key=lambda a:pre_score(q,p,a),reverse=True)
    inspected=[read_article(a) for a in allc[:40]]
    ev=evaluate(q,p,queries,inspected)

    follow=[x.strip() for x in ev.get("followups",[]) if isinstance(x,str) and x.strip()]
    used={fold(x) for x in queries};follow=[x for x in follow if fold(x) not in used][:4]
    if follow:
        extra=[]
        for qq in follow:
            found,err=wol_search(qq)
            if err:errors.append(err)
            for a in found:
                if a["url"] not in seen:seen.add(a["url"]);extra.append(a)
            time.sleep(.10)
        extra.sort(key=lambda a:pre_score(q,p,a),reverse=True)
        more=[read_article(a) for a in extra[:14]]
        if more:
            inspected+=more;ev=evaluate(q,p,queries+follow,inspected)

    return {"version":"10.0","articles":format_results(inspected,ev),
            "articles_reviewed":len(inspected),"searches_performed":len(queries)+len(follow),
            "candidate_articles_found":len(allc),"message":"" if format_results(inspected,ev) else
            "No encontramos artículos que podamos confirmar como pertinentes. Probá con otra formulación.",
            "wol_errors":list(dict.fromkeys(errors)),"source":"WOL"}

@app.get("/")
def home():return FileResponse(BASE/"index.html")
@app.get("/health")
def health():return {"ok":True,"ai_configured":bool(os.getenv("OPENAI_API_KEY")),"wol_search_enabled":True,"version":"10.0"}
@app.post("/api/search")
def search(req:SearchRequest):return research(req.question)
