import streamlit as st
import streamlit.components.v1 as components
import os, shutil, base64, time
import random
import gc
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from ingestion import get_embeddings, ingest_documents
from utils import load_documents_from_files

# ── ENV ───────────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

st.set_page_config(
    page_title="AetherRAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# LOGO 
def get_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO     = get_b64("logo.png")
LOGO_SRC = f"data:image/png;base64,{LOGO}" if LOGO else ""
AVATAR   = (
    f'<img class="ai-avatar" src="{LOGO_SRC}">'
    if LOGO_SRC else
    '<div class="ai-avatar-fb"></div>'
)

# CSS
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
h1 a, h2 a, h3 a { display: none !important; }

.stApp { background: #f5f8ff; }
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8edf5;
}
[data-testid="collapsedControl"] {
    top: 12px !important;
    left: 12px !important;
    width: 32px;
    height: 32px;
}
[data-testid="collapsedControl"] svg {
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}
[data-testid="collapsedControl"]::after {
    content: "☰";
    font-size: 22px;
    color: #4a5568;
    position: absolute;
    top: 2px;
    left: 4px;
}
textarea, input {
    caret-color: #4a7ab5 !important;
}
.app-header {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #1a1a2e;
    text-shadow:
        0 0 20px rgba(100,149,237,0.50),
        0 0 55px rgba(100,149,237,0.22);
    margin-top: -22px;
    margin-bottom: 0;
}
.chat-wrap { padding: 4px 0 90px 0; }
.msg-row {
    display: flex;
    align-items: flex-start;
    margin: 3px 0 10px 0;
}
.msg-row.user { justify-content: flex-end; }
.msg-row.ai   { justify-content: flex-start; }
.ai-avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 7px;
    flex-shrink: 0;
    margin-top: 1px;
}
.ai-avatar-fb {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6495ed, #87ceeb);
    margin-right: 7px;
    flex-shrink: 0;
}
@keyframes ai-glow-pulse {
    0%   { box-shadow: 0 0 0   rgba(106,169,255,0); }
    40%  { box-shadow: 0 0 18px rgba(106,169,255,0.65); }
    100% { box-shadow: 0 0 0   rgba(106,169,255,0); }
}
.fade-out {
    animation: fadeOut 0.35s ease forwards;
}
@keyframes fadeOut {
    from { opacity: 1; transform: scale(1); }
    to   { opacity: 0; transform: scale(0.98); }
}
.bubble {
    max-width: 60%;
    padding: 10px 14px;
    font-size: 14.5px;
    line-height: 1.65;
    word-wrap: break-word;
    white-space: pre-wrap;
    position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    transition: transform 0.15s ease;
}
.bubble:hover { transform: translateY(-2px); }
.bubble.ai.new { animation: ai-glow-pulse 0.8s ease-out; }
.bubble.user {
    background: #e8e8e8;
    color: #1a1a1a;
    border-radius: 14px 0 14px 14px;
}
.bubble.user::after {
    content: '';
    position: absolute;
    top: 0; right: -8px;
    border-left: 8px solid #e8e8e8;
    border-bottom: 8px solid transparent;
}
.bubble.ai {
    background: #d4eeff;
    color: #1a1a1a;
    border-radius: 0 14px 14px 14px;
}
.bubble.ai::before {
    content: '';
    position: absolute;
    top: 0; left: -8px;
    border-right: 8px solid #d4eeff;
    border-bottom: 8px solid transparent;
}
.tdots { display:flex; gap:5px; align-items:center; padding:2px 0; }
.tdots span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #7aaec8;
    animation: tdot 1.2s infinite ease-in-out;
}
.tdots span:nth-child(2) { animation-delay: .18s; }
.tdots span:nth-child(3) { animation-delay: .36s; }
@keyframes tdot {
    0%, 80%, 100% { transform: translateY(0); }
    40%           { transform: translateY(-6px); }
}
div.stButton > button {
    font-size: 12px;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1px solid #c8dff0;
    background: #ffffff;
    color: #2a5a8a;
    transition: all 0.15s;
    box-shadow: 0 1px 4px rgba(100,180,255,0.12);
    margin-bottom: 4px;
}
div.stButton > button:hover {
    background: #e8f4ff;
    border-color: #7ab0d8;
    box-shadow: 0 2px 8px rgba(100,180,255,0.22);
}
section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 8px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# CACHED RESOURCES 
@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    return get_embeddings()

@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(
        base_url=OLLAMA_HOST,
        model="mistral:7b-instruct-v0.3-q4_K_M",
        temperature=0.55,
        num_ctx=4096,
        num_predict=700,
    )

RAG_PROMPT = PromptTemplate(
    template="""
You are AetherRAG, a helpful AI assistant.

Answer using ONLY the provided context.

Guidelines:
- Provide clear, structured answers
- If multiple sources disagree, mention both viewpoints
- Always stay grounded in context
- Do NOT make up information

Context:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question"],
)

def load_vs():
    if os.path.exists("./faiss_index/index.faiss"):
        return FAISS.load_local(
            "./faiss_index",
            get_embeddings_cached(),
            allow_dangerous_deserialization=True,
        )
    return None

def build_chain(vs):
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=vs.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

# STREAM HANDLER 
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder  = placeholder
        self.text         = ""
        self._last_render = 0.0

    def _render(self, cursor=True):
        c    = '<span style="color:#8aaec8;font-weight:300;">|</span>' if cursor else ""
        html = (
            f'<div class="msg-row ai">{AVATAR}'
            f'<div class="bubble ai">{self.text}{c}</div></div>'
        )
        self.placeholder.markdown(html, unsafe_allow_html=True)

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        now = time.time()
        if now - self._last_render > 0.02:
            self._render(cursor=True)
            self._last_render = now

    def on_llm_end(self, *args, **kwargs):
        html = (
            f'<div class="msg-row ai">{AVATAR}'
            f'<div class="bubble ai new">{self.text}</div></div>'
        )
        self.placeholder.markdown(html, unsafe_allow_html=True)

# SESSION STATE 
if "messages"        not in st.session_state: st.session_state.messages        = []
if "vectorstore"     not in st.session_state: st.session_state.vectorstore     = load_vs()
if "ingested_files"  not in st.session_state: st.session_state.ingested_files  = set()
if "pending_query"   not in st.session_state: st.session_state.pending_query   = None
if "confirm_delete"  not in st.session_state: st.session_state.confirm_delete  = False
if "fade_delete"     not in st.session_state: st.session_state.fade_delete     = False
# uploader_key is incremented to force a fresh file_uploader widget on clear
# (Streamlit forbids setting a file_uploader value via session_state directly)
if "uploader_key"    not in st.session_state: st.session_state.uploader_key    = 0

# DEFERRED CLEAR DATA 
if st.session_state.fade_delete:
    time.sleep(0.3)

    # 1. Release FAISS
    st.session_state.vectorstore = None
    gc.collect()

    # 2. Delete FILES INSIDE, not folder
    index_path = "./faiss_index"

    if os.path.exists(index_path):
        for filename in os.listdir(index_path):
            file_path = os.path.join(index_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")

    # 3. Reset state
    st.session_state.messages       = []
    st.session_state.pending_query  = None
    st.session_state.ingested_files = set()
    st.session_state.uploader_key  += 1
    st.session_state.fade_delete    = False
    st.session_state.confirm_delete = False

    st.rerun()

# HEADER 
st.markdown('<div class="app-header">AetherRAG</div>', unsafe_allow_html=True)

# TAGLINE 
components.html("""
<style>
  * { margin:0; padding:0; }
  body { background:transparent;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  .tl { font-size:13.5px; color:#888; user-select:none; }
  #tw { color:#4a7ab5; font-weight:500; }
  .cur {
    display:inline-block; width:1.5px; height:12px;
    background:#4a7ab5; vertical-align:middle; margin-left:1px;
    animation: bl 0.85s step-end infinite;
  }
  @keyframes bl { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
<div class="tl">AI that is <span id="tw"></span><span class="cur" id="cur"></span></div>
<script>
(function(){
  const words = ['private','local','free','secure','yours'];
  const el    = document.getElementById('tw');
  const cur   = document.getElementById('cur');
  let wi=0, ci=0, del=false;

  if(sessionStorage.getItem('aether_td')){
    el.textContent = 'yours.';
    cur.style.display = 'none';
    return;
  }
  function tick(){
    const w = words[wi];
    if(!del){
      el.textContent = w.slice(0, ++ci);
      if(ci === w.length){
        if(wi === words.length - 1){
          setTimeout(()=>{
            cur.style.animation = 'none';
            cur.style.opacity   = '0';
            el.textContent      = 'yours.';
            sessionStorage.setItem('aether_td','1');
          }, 2000);
          return;
        }
        return setTimeout(()=>{ del=true; tick(); }, 750);
      }
    } else {
      el.textContent = w.slice(0, --ci);
      if(ci === 0){ del=false; wi++; }
    }
    setTimeout(tick, del ? 48 : 78);
  }
  tick();
})();
</script>
""", height=24, scrolling=False)

# SIDEBAR 
with st.sidebar:
    if LOGO:
        st.image("logo.png", width=54)
    st.markdown("### Aether")
    st.caption("Local · Private · Offline")
    st.divider()

    st.markdown("**Upload documents**")
    # key uses uploader_key counter — incrementing it remounts the widget
    # with no files, which is the only safe way to "reset" a file_uploader
    files = st.file_uploader(
        " ",
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    if files:
        new_files = set(f.name for f in files) - st.session_state.ingested_files

        if new_files:
            with st.spinner("Ingesting…"):
                docs = []

                for f in files:
                    if f.name in new_files:
                        docs.extend(load_documents_from_files([f]))

                if docs:
                    ingest_documents(docs)   # append, NOT rebuild

                    st.session_state.vectorstore = None 
                    st.session_state.ingested_files.update(new_files)

            st.success(f"✓ Added {len(new_files)} new file(s)")

    st.divider()

    if st.button("🗑  Delete chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑  Clear all data", use_container_width=True):
        st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        st.warning("⚠️ This will delete all uploaded data and embeddings.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete", use_container_width=True):
                st.session_state.fade_delete   = True
                st.session_state.confirm_delete = False
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()

    st.divider()

# BUBBLE HELPERS 
def user_bubble(text):
    return (
        f'<div class="msg-row user">'
        f'<div class="bubble user">{text}</div>'
        f'</div>'
    )

def ai_bubble(text, thinking=False, is_new=False):
    body = (
        '<div class="tdots"><span></span><span></span><span></span></div>'
        if thinking else text
    )
    cls = "bubble ai" + (" new" if is_new else "")
    return (
        f'<div class="msg-row ai">'
        f'{AVATAR}'
        f'<div class="{cls}">{body}</div>'
        f'</div>'
    )

# ONBOARDING 
if not st.session_state.messages:
    st.markdown("**Upload your documents, then try asking:**")
    cols = st.columns(2)
    for i, s in enumerate([
        "Summarize this document",
        "What are the key points?",
        "Explain this in simple terms",
        "Who are the main people mentioned?",
    ]):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.session_state.pending_query = s
                st.rerun()

# CHAT HISTORY 
fade_class = "fade-out" if st.session_state.fade_delete else ""
st.markdown(f'<div class="chat-wrap {fade_class}">', unsafe_allow_html=True)
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(user_bubble(m["content"]), unsafe_allow_html=True)
    else:
        st.markdown(ai_bubble(m["content"]), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# INPUT + INFERENCE 
query = st.chat_input("Ask something…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending_query = query
    st.rerun()

query = st.session_state.get("pending_query")

if query:
    if not isinstance(query, str) or not query.strip():
        st.session_state.pending_query = None
        st.stop()

    st.session_state.pending_query = None

    # ── CONTEXT-AWARE GREETING HANDLER ──
    greetings = ["hi", "hello", "hey", "yo", "hola", "heya"]
    q = query.lower().strip()
    q_clean = q.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    tokens = q_clean.split()
    # Detect greeting intent only if it appears at the START
    if tokens and tokens[0] in greetings:
        replies = [
            "Hey there! Ready to begin?",
            "Hey! How are you today?",
            "Hey there, nice to see you.",
            "Hey there — how can I help you today?",
            "Hey! What's on your mind?",
            "Greetings! How can I assist you today?",
            "Hey! What's up?",
            "Hey! How's it going?",
            "Hey! I am Aether, your personal RAG assistant. How can I help you today?",
            "Hey there! Aether here — how can I help today?",
        ]

        # Only greeting → respond & stop
        if len(tokens) == 1:
            st.session_state.messages.append({
                "role": "assistant",
                "content": random.choice(replies)
            })
            st.rerun()
        # Greeting + actual question → respond AND continue to RAG
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": random.choice(replies)
            })
        # DO NOT rerun → allow RAG to continue

        # ── ACKNOWLEDGEMENT HANDLER ──
    acks = ["ok", "okay", "hmm", "alright", "fine", "cool"]
    q = query.lower().strip().replace(",", "").replace(".", "")
    if len(q.split()) <= 2 and any(word in acks for word in q.split()):
        replies = [
            "Got it. What else would you like to know?",
            "Alright 👍 What should we explore next?",
            "Cool. Want me to explain something else?",
            "Okay — go ahead, ask your next question.",
            "Hmm noted. What’s next?",
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "content": random.choice(replies)
        })
        st.rerun()

    # RAG HANDLER 
    elif True:

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = load_vs()

        if not st.session_state.vectorstore:
            st.info("⬅️ Upload a document from the sidebar to get started.")
            st.stop()

        #    similarity_search_with_relevance_scores normalises correctly for L2/cosine
        #    unlike similarity_search_with_score which returns raw L2 distances
        docs_with_scores = st.session_state.vectorstore.similarity_search_with_relevance_scores(
            query, k=5
        )
        confidences = [score for _, score in docs_with_scores]
        max_conf    = max(confidences) if confidences else 0.0

        ai_slot = st.empty()
        ai_slot.markdown(ai_bubble("", thinking=True), unsafe_allow_html=True)

        if max_conf < 0.35:
            answer = (
                "I couldn't find relevant information in the uploaded documents. "
                "Please try asking something else or upload more documents."
            )
            ai_slot.markdown(ai_bubble(answer, is_new=True), unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # 2. Run chain
        handler       = StreamHandler(ai_slot)
        llm           = get_llm()
        llm.callbacks = [handler]

        chain  = build_chain(st.session_state.vectorstore)
        result = chain.invoke({"query": query})

        answer = handler.text or result.get("result", "")

        sources     = result.get("source_documents", [])
        source_text = ""
        if sources:
            seen = set()
            for doc in sources:
                src       = doc.metadata.get("source", "Unknown")
                highlight = (doc.page_content or "").strip().replace("\n", " ")[:200]
                if src not in seen:
                    source_text += (
                        f'<div style="margin-top:8px;font-size:12px;color:#555;">'
                        f'📄 <b>{src}</b><br>'
                        f'<span style="background:#fff3b0;padding:2px 5px;border-radius:4px;">'
                        f'{highlight}…</span></div>'
                    )
                    seen.add(src)

        pct = round(max_conf * 100, 1)
        if max_conf >= 0.70:
            conf_label = f"🟢 {pct}% confident"
        elif max_conf >= 0.50:
            conf_label = f"🟡 {pct}% low confidence"
        else:
            conf_label = f"🔴 {pct}% very low confidence"

        confidence_text = (
            f'<div style="font-size:11px;color:#888;margin-top:6px;">'
            f'{conf_label}</div>'
        )

        full_answer = answer + source_text + confidence_text

        # 5. Final render + persist
        ai_slot.markdown(ai_bubble(full_answer, is_new=True), unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

    else:
        st.info("⬅️ Upload a document from the sidebar to get started.")