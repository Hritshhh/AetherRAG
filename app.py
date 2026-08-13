import streamlit as st
import streamlit.components.v1 as components
import os, base64, gc, random, time
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from ingestion import get_embeddings, ingest_documents
from utils import load_documents_from_files

# ── CONFIG & CONSTANTS ────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CONFIDENCE_THRESHOLD = 0.35
K_DOC_QUERY = 15
K_SHORT_QUERY = 5
K_DEFAULT = 3

st.set_page_config(
    page_title="AetherRAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── LOGO ──────────────────────────────────────────────────────────────────────
def get_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO = get_b64("assets/logo.png")
LOGO_SRC = f"data:image/png;base64,{LOGO}" if LOGO else ""
AVATAR = (
    f'<img class="ai-avatar" src="{LOGO_SRC}">'
    if LOGO_SRC else
    '<div class="ai-avatar-fb"></div>'
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden;}
h1 a,h2 a,h3 a{display:none!important;}
.stApp{background:#f5f8ff;}
section[data-testid="stSidebar"]{
background:#fff;
border-right:1px solid #e8edf5;
}
[data-testid="collapsedControl"]{
top:12px!important;
left:12px!important;
width:32px;
height:32px;
}
[data-testid="collapsedControl"] svg{
display:none!important;
}
[data-testid="collapsedControl"]::after{
content:"☰";
font-size:22px;
color:#4a5568;
position:absolute;
top:2px;
left:4px;
}
textarea,input{
caret-color:#4a7ab5!important;
}
.app-header{
font-size:42px;
font-weight:800;
letter-spacing:-1.5px;
color:#1a1a2e;
text-shadow:
0 0 20px rgba(100,149,237,.5),
0 0 55px rgba(100,149,237,.22);
margin-top:-22px;
margin-bottom:0;
}
.chat-wrap{
padding:4px 0 90px;
}
.msg-row{
display:flex;
align-items:flex-start;
margin:3px 0 10px;
}
.msg-row.user{justify-content:flex-end;}
.msg-row.ai{justify-content:flex-start;}

.ai-avatar{
width:30px;
height:30px;
border-radius:50%;
object-fit:cover;
margin:1px 7px 0 0;
flex-shrink:0;
}

.ai-avatar-fb{
width:30px;
height:30px;
border-radius:50%;
background:linear-gradient(135deg,#6495ed,#87ceeb);
margin-right:7px;
flex-shrink:0;
}

@keyframes ai-glow-pulse{
0%{box-shadow:0 0 0 rgba(106,169,255,0);}
40%{box-shadow:0 0 18px rgba(106,169,255,.65);}
100%{box-shadow:0 0 0 rgba(106,169,255,0);}
}

.fade-out{
animation:fadeOut .35s ease forwards;
}

@keyframes fadeOut{
from{opacity:1;transform:scale(1);}
to{opacity:0;transform:scale(.98);}
}

.bubble{
max-width:60%;
padding:10px 14px;
font-size:14.5px;
line-height:1.65;
white-space:pre-wrap;
word-wrap:break-word;
position:relative;
box-shadow:0 1px 4px rgba(0,0,0,.08);
transition:transform .15s;
}

.bubble:hover{
transform:translateY(-2px);
}

.bubble.ai.new{
animation:ai-glow-pulse .8s ease-out;
}

.bubble.user{
background:#e8e8e8;
color:#1a1a1a;
border-radius:14px 0 14px 14px;
}

.bubble.user::after{
content:"";
position:absolute;
top:0;
right:-8px;
border-left:8px solid #e8e8e8;
border-bottom:8px solid transparent;
}

.bubble.ai{
background:#d4eeff;
color:#1a1a1a;
border-radius:0 14px 14px 14px;
}

.bubble.ai::before{
content:"";
position:absolute;
top:0;
left:-8px;
border-right:8px solid #d4eeff;
border-bottom:8px solid transparent;
}

.tdots{
display:flex;
gap:5px;
align-items:center;
padding:2px 0;
}

.tdots span{
width:7px;
height:7px;
border-radius:50%;
background:#7aaec8;
animation:tdot 1.2s infinite ease-in-out;
}

.tdots span:nth-child(2){animation-delay:.18s;}
.tdots span:nth-child(3){animation-delay:.36s;}

@keyframes tdot{
0%,80%,100%{transform:translateY(0);}
40%{transform:translateY(-6px);}
}
div.stButton>button{
font-size:12px;
padding:6px 16px;
border-radius:20px;
border:1px solid #c8dff0;
background:#fff;
color:#2a5a8a;
transition:.15s;
box-shadow:0 1px 4px rgba(100,180,255,.12);
margin-bottom:4px;
}

div.stButton>button:hover{
background:#e8f4ff;
border-color:#7ab0d8;
box-shadow:0 2px 8px rgba(100,180,255,.22);
}

section[data-testid="stSidebar"] div.stButton>button{
width:100%;
text-align:left;
border-radius:8px;
margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)


# ── LLM & CACHED RESOURCES ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    return get_embeddings()


def get_llm(callbacks=None):
    return OllamaLLM(
        base_url=OLLAMA_HOST,
        model="mistral:7b-instruct-v0.3-q4_K_M",
        temperature=0.6,
        num_ctx=4096,
        num_predict=700,
        callbacks=callbacks or []
    )


RAG_PROMPT = PromptTemplate(
    template="""You are AetherRAG, a trustworthy retrieval-augmented AI assistant.

Use the provided context as the primary source of truth.

Rules:
- Never invent information.
- If context is insufficient, explicitly say so.
- Combine information across chunks when possible.
- If documents disagree, explain both viewpoints.
- Write naturally and clearly.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

GENERAL_PROMPT = """You are Aether, a helpful AI assistant.

Answer naturally using your own knowledge.
If you are uncertain, say so.

Question:
{question}

Answer:
"""


def load_vs():
    if not os.path.exists("./faiss_index/index.faiss"):
        return None
    return FAISS.load_local(
        "./faiss_index",
        get_embeddings_cached(),
        allow_dangerous_deserialization=True,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(llm):
    # Pure LCEL Architecture: Bypasses legacy langchain.chains entirely
    return (
            {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
    )


# ── QUERY CLASSIFICATION & RETRIEVAL ──────────────────────────────────────────
DOCUMENT_INTENTS = {
    "summarize", "summary", "overview", "tldr", "tl;dr",
    "key point", "key points", "main point", "main points",
    "explain the document", "explain this document", "explain this",
    "simplify", "simple terms", "this document", "the document",
    "pdf", "file", "context", "who is mentioned", "what is this document about",
    "what does this document discuss", "document summary", "summarize the document",
    "summarize this"
}


def is_document_query(query: str) -> bool:
    q = query.lower()
    return "document" in q or "pdf" in q or any(x in q for x in DOCUMENT_INTENTS)

def retrieve_documents(vs, query: str):
    doc_query = is_document_query(query)
    k = K_DOC_QUERY if doc_query else (K_SHORT_QUERY if len(query.split()) <= 3 else K_DEFAULT)
    docs_with_scores = vs.similarity_search_with_relevance_scores(query, k=k)
    max_conf = max((s for _, s in docs_with_scores), default=0.0)
    return docs_with_scores, max_conf, doc_query

def run_general_llm(query: str, llm):
    return llm.invoke(GENERAL_PROMPT.format(question=query)).strip()


def get_confidence_badge(conf_float):
    pct = int(conf_float * 100)
    if pct >= 60:
        color = "#28a745"  # green
    elif pct >= 35:
        color = "#e5a800"  # yellow
    else:
        color = "#dc3545"  # red
    return f"<span style='color:{color}; font-weight:600;'>{pct}%</span>"


# ── STREAM HANDLER ────────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
        self.last_render = 0

    def render(self, cursor=True):
        cur = '<span style="color:#8aaec8;font-weight:300;">|</span>' if cursor else ""
        self.placeholder.markdown(
            f'<div class="msg-row ai">{AVATAR}<div class="bubble ai">{self.text}{cur}</div></div>',
            unsafe_allow_html=True,
        )

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        now = time.time()
        if now - self.last_render > 0.02:
            self.render()
            self.last_render = now

    def on_llm_end(self, *args, **kwargs):
        self.placeholder.markdown(
            f'<div class="msg-row ai">{AVATAR}<div class="bubble ai new">{self.text}</div></div>',
            unsafe_allow_html=True,
        )


# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vectorstore": load_vs(),
    "ingested_files": set(),
    "pending_query": None,
    "confirm_delete": False,
    "fade_delete": False,
    "uploader_key": 0,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CLEAR DATA ────────────────────────────────────────────────────────────────
if st.session_state.fade_delete:
    time.sleep(.3)
    st.session_state.vectorstore = None
    gc.collect()
    index = "./faiss_index"
    if os.path.exists(index):
        for f in os.listdir(index):
            p = os.path.join(index, f)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
    st.session_state.messages = []
    st.session_state.ingested_files = set()
    st.session_state.pending_query = None
    st.session_state.confirm_delete = False
    st.session_state.fade_delete = False
    st.session_state.uploader_key += 1
    st.rerun()

# ── HEADER & TAGLINE ──────────────────────────────────────────────────────────
st.markdown('<div class="app-header">AetherRAG</div>', unsafe_allow_html=True)
components.html("""
<style>
*{margin:0;padding:0}
body{
background:transparent;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
.tl{
font-size:13.5px;
color:#888;
user-select:none;
}
#tw{
color:#4a7ab5;
font-weight:500;
}
.cur{
display:inline-block;
width:1.5px;
height:12px;
background:#4a7ab5;
vertical-align:middle;
margin-left:1px;
animation:bl .85s step-end infinite;
}
@keyframes bl{
0%,100%{opacity:1}
50%{opacity:0}
}
</style>
<div class="tl">AI that is <span id="tw"></span><span class="cur" id="cur"></span></div>
<script>
(()=>{
const words=["private","local","free","secure","yours"];
const tw=document.getElementById("tw");
const cur=document.getElementById("cur");
let wi=0;
let ci=0;
let deleting=false;
if(sessionStorage.getItem("aether_tagline")){
    tw.textContent="yours.";
    cur.style.display="none";
    return;
}
function tick(){
    const w=words[wi];
    if(!deleting){
        tw.textContent=w.slice(0,++ci);
        if(ci===w.length){
            if(wi===words.length-1){
                setTimeout(()=>{
                    tw.textContent="yours.";
                    cur.style.display="none";
                    sessionStorage.setItem("aether_tagline","1");
                },1800);
                return;
            }
            return setTimeout(()=>{
                deleting=true;
                tick();
            },700);
        }
    }else{
        tw.textContent=w.slice(0,--ci);
        if(ci===0){
            deleting=false;
            wi++;
        }
    }
    setTimeout(tick,deleting?45:75);
}
tick();
})();
</script>
""", height=24, scrolling=False)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO:
        st.image("assets/logo.png", width=54)
    st.markdown("### Aether")
    st.caption("Local · Private · Offline")
    st.divider()
    st.markdown("**Upload documents**")
    files = st.file_uploader(
        "",
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"upload_{st.session_state.uploader_key}",
    )
    if files:
        new_files = {f.name for f in files} - st.session_state.ingested_files
        if new_files:
            with st.spinner("Ingesting documents..."):
                docs = []
                for f in files:
                    if f.name in new_files:
                        docs.extend(load_documents_from_files([f]))
                if docs:
                    ingest_documents(docs)
                    st.session_state.vectorstore = None
                    st.session_state.ingested_files.update(new_files)
            st.success(f"Added {len(new_files)} file(s).")
    st.divider()
    if st.button("🗑 Delete Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("🗑 Clear All Data", use_container_width=True):
        st.session_state.confirm_delete = True
    if st.session_state.confirm_delete:
        st.warning("This will permanently remove uploaded documents and embeddings.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Delete", use_container_width=True):
                st.session_state.confirm_delete = False
                st.session_state.fade_delete = True
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()


# ── CHAT HELPERS ──────────────────────────────────────────────────────────────
def user_bubble(text):
    return f'<div class="msg-row user"><div class="bubble user">{text}</div></div>'


def ai_bubble(text="", thinking=False, new=False):
    body = '<div class="tdots"><span></span><span></span><span></span></div>' if thinking else text
    cls = "bubble ai new" if new else "bubble ai"
    return f'<div class="msg-row ai">{AVATAR}<div class="{cls}">{body}</div></div>'


# ── ONBOARDING ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Upload your documents, then try asking:**")
    suggestions = [
        "Summarize the uploaded document",
        "What are the key points in this document?",
        "Explain the document in simple terms",
        "Who are the main people mentioned in the document?",
    ]
    c1, c2 = st.columns(2)
    for i, s in enumerate(suggestions):
        with (c1 if i % 2 == 0 else c2):
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.session_state.pending_query = s
                st.rerun()

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
fade = " fade-out" if st.session_state.fade_delete else ""
st.markdown(f'<div class="chat-wrap{fade}">', unsafe_allow_html=True)
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(user_bubble(m["content"]), unsafe_allow_html=True)
    else:
        st.markdown(ai_bubble(m["content"]), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_query = prompt
    st.rerun()

query = st.session_state.pending_query
if not query:
    st.stop()
st.session_state.pending_query = None

# ── GREETING & ACKNOWLEDGEMENT HANDLERS ───────────────────────────────────────
clean_query = query.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "")
words = clean_query.split()

greetings = {"hi", "hello", "hey", "heya", "hiya", "yo", "hola", "greetings", "sup", "howdy"}
if len(words) <= 2 and any(w in greetings for w in words):
    reply = random.choice([
        "Hey! How can I help you today?",
        "Hello! What would you like to know?",
        "Hi there! Ready when you are.",
        "Hey! Ask me anything.",
        "Hello 👋",
    ])
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

acks = {"ok", "okay", "alright", "fine", "cool", "thanks", "thank", "thankyou", "thx", "gotcha", "understood"}
if len(words) <= 2 and any(w in acks for w in words):
    reply = random.choice([
        "You're welcome!",
        "Glad I could help.",
        "Anytime.",
        "Happy to help.",
        "What would you like to ask next?",
    ])
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ── PREPARE HYBRID RAG ────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vs()

ai_placeholder = st.empty()
ai_placeholder.markdown(ai_bubble(thinking=True), unsafe_allow_html=True)

handler = StreamHandler(ai_placeholder)
llm = get_llm(callbacks=[handler])

# ── NO DOCUMENTS → GENERAL LLM ────────────────────────────────────────────────
if not st.session_state.vectorstore:
    answer = run_general_llm(query, llm)
    answer += (
        "<br><span style='font-size:11px;color:#888;'>"
        "Answered using general knowledge because no documents are uploaded."
        "</span>"
    )
    ai_placeholder.markdown(ai_bubble(answer, new=True), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── RETRIEVAL & CONFIDENCE CHECK ──────────────────────────────────────────────
docs_with_scores, max_conf, is_doc_query = retrieve_documents(st.session_state.vectorstore, query)

# ── LOW CONFIDENCE → GENERAL LLM ──────────────────────────────────────────────
if not is_doc_query and max_conf < CONFIDENCE_THRESHOLD:
    answer = run_general_llm(query, llm)

    # Hide the confidence footer for purely conversational/short queries to avoid confusion
    if len(words) > 3:
        badge = get_confidence_badge(max_conf)
        answer += (
            f"<br><span style='font-size:11px;color:#888;'>"
            f"Answered using general knowledge (Retrieval confidence: {badge})."
            f"</span>"
        )

    ai_placeholder.markdown(ai_bubble(answer, new=True), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── HIGH CONFIDENCE OR DOCUMENT QUERY → RAG ───────────────────────────────────
docs = [doc for doc, score in docs_with_scores]
chain = build_chain(llm)

# Execute LCEL chain
answer = chain.invoke({
    "context": docs,
    "question": query
}).strip()

if not answer:
    answer = "I couldn't generate a response from the uploaded documents."

badge = get_confidence_badge(max_conf)
answer += f"<br><span style='font-size:11px;color:#888;'>Answered using document context (Confidence: {badge}).</span>"

ai_placeholder.markdown(ai_bubble(answer, new=True), unsafe_allow_html=True)
st.session_state.messages.append({"role": "assistant", "content": answer})

# ── SOURCES ───────────────────────────────────────────────────────────────────
if docs_with_scores:
    with st.expander("📚 Sources", expanded=False):
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page")

            title = f"**{i}. {src}"
            if page is not None:
                title += f" (Page {page + 1})"
            title += f" — Relevance: {int(score * 100)}%**"

            st.markdown(title)

            preview = doc.page_content.strip()
            if len(preview) > 600:
                preview = preview[:600] + "..."

            # Use blockquote for highlighted snippet appearance
            st.markdown(f"> {preview}")

st.rerun()