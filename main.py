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
HTTP.headers.update({"User-Agent":"Mozilla/5.0 (compatible; WOLInteligente/11.1)","Accept-Language":"es-AR,es;q=0.9"})
app=FastAPI(title="WOL Inteligente v11.1")

class SearchRequest(BaseModel):
    question:str=Field(min_length=3,max_length=1500)

TYPE_PROMPT="""Clasifica la necesidad sin responderla. Tipos: procedure, advice, problem, emotional, concept, doctrinal, historical, person_role, situation, thematic.
procedure=como hacer algo concreto; advice=consejo; problem=dificultad; emotional=sentimientos/emociones/animo; concept=significado; doctrinal=pregunta doctrinal; historical=historia; person_role=ayudar/tratar con persona o rol; situation=situacion; thematic=tema amplio.
No fuerces accion+objeto en emotional ni thematic."""
TYPE_SCHEMA={"type":"object","properties":{"type":{"type":"string","enum":["procedure","advice","problem","emotional","concept","doctrinal","historical","person_role","situation","thematic"]},"reason":{"type":"string"}},"required":["type","reason"],"additionalProperties":False}

PLAN_PROMPT="""Sos investigador de WOL en espanol. Modela la necesidad y crea consultas concretas. Separa accion, objeto, contexto, rol, problema, objetivo y conceptos. No supongas La Atalaya salvo que se mencione. Para emotional/thematic prioriza rutas tematicas; para procedure conserva accion+objeto+contexto. No inventes terminologia WOL: se descubre leyendo documentos reales."""
PLAN_SCHEMA={"type":"object","properties":{"intent":{"type":"string"},"action":{"type":"string"},"object":{"type":"string"},"context":{"type":"string"},"role":{"type":"string"},"goal":{"type":"string"},"concepts":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":12},"must_match":{"type":"array","items":{"type":"string"},"maxItems":10},"avoid":{"type":"array","items":{"type":"string"},"maxItems":10},"queries":{"type":"array","items":{"type":"string"},"minItems":6,"maxItems":16}},"required":["intent","action","object","context","role","goal","concepts","must_match","avoid","queries"],"additionalProperties":False}
VOCAB_PROMPT="""Lee documentos REALES de WOL. Extrae solo vocabulario/expresiones presentes y utiles para nuevas rutas. Para emotional/thematic prioriza palabras que representen aspectos del tema aunque difieran de las palabras del usuario. Descarta palabras accidentales. Construye nuevas consultas con vocabulario confirmado."""
VOCAB_SCHEMA={"type":"object","properties":{"confirmed_terms":{"type":"array","items":{"type":"string"},"maxItems":24},"useful_phrases":{"type":"array","items":{"type":"string"},"maxItems":14},"new_queries":{"type":"array","items":{"type":"string"},"maxItems":14},"gaps":{"type":"array","items":{"type":"string"},"maxItems":8}},"required":["confirmed_terms","useful_phrases","new_queries","gaps"],"additionalProperties":False}
JUDGE_PROMPT="""Juzga pertinencia de articulos REALES. No respondas ni premies coincidencias de palabras.
procedure: orientacion concreta sobre accion, objeto y contexto cuando existan.
advice/problem/person_role/situation: tratar la situacion o ayuda concreta.
emotional: tratar sustancialmente sentimientos/estado emocional y contexto; puede usar vocabulario distinto descubierto en WOL y no exige accion.
thematic: tratar sustancialmente el tema y no exige accion.
concept/doctrinal/historical: abordar realmente la pregunta.
Mencion pasajera, articulo generico, contexto distinto o coincidencia superficial NO alcanza. Selecciona solo score 4 o 5 con evidencia concreta. Followups solo para huecos reales."""
JUDGE_SCHEMA={"type":"object","properties":{"selected":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"score":{"type":"integer","minimum":4,"maximum":5},"coverage":{"type":"string"},"evidence":{"type":"string"},"doubt":{"type":"string"}},"required":["id","score","coverage","evidence","doubt"],"additionalProperties":False},"maxItems":24},"missing":{"type":"array","items":{"type":"string"},"maxItems":10},"followups":{"type":"array","items":{"type":"string"},"maxItems":8}},"required":["selected","missing","followups"],"additionalProperties":False}
AUDIT_PROMPT="""Auditor adversarial: intenta descartar cada articulo. Conserva solo evidencia sustancial. No penalices vocabulario distinto en consultas emotional/thematic si el texto trata el asunto. Para procedure exige accion, objeto y contexto. Penaliza menciones pasajeras, contexto diferente y resultados genericos."""
AUDIT_SCHEMA={"type":"object","properties":{"survivors":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"reason":{"type":"string"}},"required":["id","reason"],"additionalProperties":False},"maxItems":18},"rejected":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},"reason":{"type":"string"}},"required":["id","reason"],"additionalProperties":False},"maxItems":24},"need_more_research":{"type":"boolean"},"gaps":{"type":"array","items":{"type":"string"},"maxItems":8}},"required":["survivors","rejected","need_more_research","gaps"],"additionalProperties":False}
FINAL_PROMPT="Elige el conjunto minimo que satisface la necesidad. Solo IDs auditados. Elimina redundancias y articulos genericos. No rellenes cantidad."
FINAL_SCHEMA={"type":"object","properties":{"final_ids":{"type":"array","items":{"type":"integer"},"maxItems":12}},"required":["final_ids"],"additionalProperties":False}

def ai(instructions,payload,schema,name,tokens=3000):
    if not os.getenv("OPENAI_API_KEY"): return None
    try:
        from openai import OpenAI
        r=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),timeout=75,max_retries=1).responses.create(model=os.getenv("OPENAI_MODEL","gpt-6"),instructions=instructions,input=json.dumps(payload,ensure_ascii=False),text={"format":{"type":"json_schema","name":name,"strict":True,"schema":schema}},max_output_tokens=tokens)
        return json.loads(r.output_text)
    except Exception as e:
        print(f"AI {name} error: {e!r}");return None

def fold(s):
    s=unicodedata.normalize("NFD",str(s).lower())
    return "".join(c for c in s if unicodedata.category(c)!="Mn")
STOP={"como","para","sobre","cuando","donde","quiero","puedo","puede","hacer","algo","alguien","esto","esta","este","desde","tiene","tienen","informacion","buscar","encontrar","articulos","biblioteca","linea","wol","dice","decir","hay","que","los","las","una","unos","del","por","con"}
def toks(s): return {x for x in re.findall(r"[a-záéíóúñü]{3,}",fold(s)) if x not in STOP}
def fallback_type(q):
    f=fold(q)
    if any(x in f for x in ("sentimiento","emocion","animo","soledad","alegria","tristeza","desanimo","felicidad")): return "emotional"
    if re.search(r"\\bcomo\\b",f): return "procedure"
    return "thematic" if len(toks(q))<=3 else "situation"
def classify(q): return ai(TYPE_PROMPT,{"question":q},TYPE_SCHEMA,"wol_type_v11_1",1200) or {"type":fallback_type(q)}
def fallback_plan(q,kind):
    t=list(toks(q))[:9];qs=[q]
    if kind in ("emotional","thematic"): qs += [" ".join(t[:5])," ".join(t[:4])+" sentimientos"," ".join(t[:4])+" emociones"," ".join(t[:4])+" alegria"," ".join(t[:4])+" soledad"," ".join(t[:4])+" animo"]
    else: qs += [" ".join(t[:5])," ".join(t[:4])+" consejos"," ".join(t[:4])+" ayudar"," ".join(t[:4])+" preparacion"]
    return {"intent":q,"action":"","object":"","context":"","role":"","goal":"","concepts":t,"must_match":t[:5],"avoid":[],"queries":qs}
def plan(q,kind):
    p=ai(PLAN_PROMPT,{"question":q,"query_type":kind},PLAN_SCHEMA,"wol_plan_v11_1",3000) or fallback_plan(q,kind)
    out=[];seen=set()
    for x in p.get("queries",[]):
        if isinstance(x,str) and x.strip() and fold(x) not in seen: seen.add(fold(x));out.append(x.strip())
    p["queries"]=out[:16];return p
def search_url(q): return SEARCH+"?"+urlencode({"q":q,"r":"occ","st":"e"})
def txt(n,limit=3000): return " ".join(n.get_text(" ",strip=True).split())[:limit] if n else ""
def valid_url(u):
    p=urlparse(u);return p.scheme=="https" and p.netloc.lower()=="wol.jw.org" and "/wol/d/" in p.path
def wol_search(q,limit=10):
    try:
        r=HTTP.get(search_url(q),timeout=25,allow_redirects=True);r.raise_for_status();s=BeautifulSoup(r.text,"html.parser");out=[];seen=set()
        for a in s.find_all("a",href=True):
            h=a.get("href","").strip()
            if "/wol/d/" not in h: continue
            u=urljoin(ORIGIN,h).split("#")[0]
            if not valid_url(u) or u in seen: continue
            title=txt(a.find(["h1","h2","h3","h4"]) or a,360)
            if len(title)<5: continue
            box=a.find_parent(["li","article"])
            out.append({"title":title,"url":u,"context":txt(box,1800) if box else title,"query":q});seen.add(u)
            if len(out)>=limit: break
        print(f"WOL {q!r}: {len(out)}");return out,None
    except Exception as e:
        print(f"WOL search error {q!r}: {e!r}");return [],f"No se pudo consultar WOL: {e}"
def publication(s):
    for sel in [".publicationName",".pubName",".publication","[class*='publicationName']","[class*='pubName']",".documentPublication"]:
        v=txt(s.select_one(sel),180)
        if v:return v
    return ""
def read_article(a):
    try:
        r=HTTP.get(a["url"],timeout=22);r.raise_for_status();s=BeautifulSoup(r.text,"html.parser")
        for n in s.select("script,style,nav,footer,header,form,.toolbar,.navigation,.publicationNavigation,.sidebar,.relatedContent,.breadcrumb"):n.decompose()
        title=txt(s.select_one("h1,h2.documentTitle,.documentTitle"),300) or a["title"]
        body=s.select_one("#article") or s.select_one("#bodyContent") or s.select_one(".bodyTxt") or s.select_one(".document") or s.select_one("main") or s.body
        parts=[]
        if body:
            for n in body.select("p,h2,h3,li"):
                v=txt(n,1200)
                if len(v)>=35:parts.append(v)
                if sum(map(len,parts))>=7500:break
        return {**a,"title":title,"publication":publication(s),"excerpt":" ".join(parts)[:7500] or a["context"][:1800]}
    except Exception as e:
        print(f"WOL article error: {e!r}");return {**a,"publication":"","excerpt":a["context"][:1800]}
def docs(arts): return [{"id":i,"title":a["title"],"found_by":a.get("query",""),"context":a.get("context","")[:1400],"excerpt":a.get("excerpt","")[:5200]} for i,a in enumerate(arts)]
def pre_rank(p,a):
    terms=set()
    for k in ("concepts","must_match"):
        for x in p.get(k,[]):terms|=toks(x)
    return 8*len(terms&toks(a.get("title","")))+2*len(terms&toks(a.get("context","")))
def search_many(qs,seen,errors):
    out=[]
    for q in qs:
        found,err=wol_search(q)
        if err:errors.append(err)
        for a in found:
            if a["url"] not in seen:seen.add(a["url"]);out.append(a)
        time.sleep(.06)
    return out
def choose_results(arts,audit,final):
    survivor_ids={x.get("id") for x in (audit or {}).get("survivors",[]) if isinstance(x,dict)}
    ids=[i for i in (final or {}).get("final_ids",[]) if isinstance(i,int) and i in survivor_ids]
    if final is None: ids=list(survivor_ids)[:8]
    out=[];seen=set()
    for i in ids:
        if 0<=i<len(arts) and arts[i]["url"] not in seen:
            seen.add(arts[i]["url"]);a=arts[i];out.append({"title":a["title"],"url":a["url"],"publication":a.get("publication","")})
    return out[:12]
def research(q):
    kind=classify(q).get("type",fallback_type(q));p=plan(q,kind);errors=[];seen=set();queries=p["queries"][:16]
    cand=search_many(queries,seen,errors);cand.sort(key=lambda a:pre_rank(p,a),reverse=True);inspected=[read_article(a) for a in cand[:50]]
    vocab=ai(VOCAB_PROMPT,{"question":q,"query_type":kind,"research_map":p,"documents":docs(inspected)},VOCAB_SCHEMA,"wol_vocab_v11_1",3000) if inspected else None
    used={fold(x) for x in queries}
    vq=[x.strip() for x in (vocab or {}).get("new_queries",[]) if isinstance(x,str) and x.strip() and fold(x) not in used][:14]
    if vq:
        queries+=vq;extra=search_many(vq,seen,errors);extra.sort(key=lambda a:pre_rank(p,a),reverse=True);inspected += [read_article(a) for a in extra[:32]]
    judged=ai(JUDGE_PROMPT,{"question":q,"query_type":kind,"research_map":p,"queries_used":queries,"articles":docs(inspected)},JUDGE_SCHEMA,"wol_judge_v11_1",4800) if inspected else None
    selected=(judged or {}).get("selected",[])
    used={fold(x) for x in queries}
    follow=[x.strip() for x in (judged or {}).get("followups",[]) if isinstance(x,str) and x.strip() and fold(x) not in used][:7]
    if follow:
        queries+=follow;extra=search_many(follow,seen,errors);extra.sort(key=lambda a:pre_rank(p,a),reverse=True);inspected += [read_article(a) for a in extra[:28]]
        judged=ai(JUDGE_PROMPT,{"question":q,"query_type":kind,"research_map":p,"queries_used":queries,"articles":docs(inspected)},JUDGE_SCHEMA,"wol_judge_v11_1b",4800) or judged
        selected=(judged or {}).get("selected",[])
    chosen=[]
    for x in selected:
        i=x.get("id") if isinstance(x,dict) else None
        if isinstance(i,int) and 0<=i<len(inspected):chosen.append({"id":i,"title":inspected[i]["title"],"excerpt":inspected[i].get("excerpt","")[:7000]})
    audit=ai(AUDIT_PROMPT,{"question":q,"query_type":kind,"research_map":p,"selected_articles":chosen},AUDIT_SCHEMA,"wol_audit_v11_1",4000) if chosen else {"survivors":[],"rejected":[],"need_more_research":False,"gaps":[]}
    if audit.get("need_more_research") and len(queries)<34:
        used={fold(x) for x in queries};gaps=[x.strip() for x in audit.get("gaps",[]) if isinstance(x,str) and x.strip() and fold(x) not in used][:6]
        if gaps:
            queries+=gaps;extra=search_many(gaps,seen,errors);extra.sort(key=lambda a:pre_rank(p,a),reverse=True);inspected += [read_article(a) for a in extra[:24]]
            judged=ai(JUDGE_PROMPT,{"question":q,"query_type":kind,"research_map":p,"queries_used":queries,"articles":docs(inspected)},JUDGE_SCHEMA,"wol_judge_v11_1c",4800) or judged
            selected=(judged or {}).get("selected",[])
            chosen=[{"id":x["id"],"title":inspected[x["id"]]["title"],"excerpt":inspected[x["id"]].get("excerpt","")[:7000]} for x in selected if isinstance(x,dict) and isinstance(x.get("id"),int) and 0<=x["id"]<len(inspected)]
            audit=ai(AUDIT_PROMPT,{"question":q,"query_type":kind,"research_map":p,"selected_articles":chosen},AUDIT_SCHEMA,"wol_audit_v11_1b",4000) if chosen else audit
    survivors=audit.get("survivors",[])
    final=ai(FINAL_PROMPT,{"question":q,"query_type":kind,"survivors":survivors,"articles":docs(inspected)},FINAL_SCHEMA,"wol_final_v11_1",2200) if survivors else {"final_ids":[]}
    results=choose_results(inspected,audit,final)
    return {"version":"11.1","articles":results,"articles_reviewed":len(inspected),"searches_performed":len(queries),"candidate_articles_found":len(seen),"message":"" if results else "No encontramos articulos que hayan superado la investigacion y auditoria de pertinencia. Proba con otra formulacion.","wol_errors":list(dict.fromkeys(errors)),"source":"WOL"}

@app.get("/")
def home(): return FileResponse(BASE/"index.html")
@app.get("/health")
def health(): return {"ok":True,"ai_configured":bool(os.getenv("OPENAI_API_KEY")),"wol_search_enabled":True,"version":"11.1"}
@app.post("/api/search")
def search(request:SearchRequest): return research(request.question)
