from __future__ import annotations
import os
from pathlib import Path
from typing import Dict,List

from fastapi import FastAPI, File,UploadFile,HTTPException,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.document_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import AIMessage,HumanMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException


app = FastAPI(title="MultiDocChat",version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
template_dir = BASE_DIR / "templates"
app.mount("/static",StaticFiles(directory=str(static_dir)),name="static")
templates= Jinja2Templates(directory=str(template_dir))


SESSION: Dict[str, List[dict]] = {}


class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()"""
    def __init__(self,uf:UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer : str


@app.get("/health")
def health() -> Dict[str,str]:
    return {"status":"ok"}


@app.get("/",response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request":request})


@app.post("/upload",response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) ->UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    try:
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        ingestor.build_retriever(wrapped_files)

        SESSION[session_id] = []

        return UploadResponse(session_id=session_id, indexed=True, message="Indexing completed with mmr")

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")
    


@app.post("/chat",response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id        
    message = req.message.strip()

    if not session_id or session_id not in SESSION:
        raise HTTPException(status_code=400, detail="Invalid or expired session. re-upload documents")
    
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    

    try:
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path
        )

        simple = SESSION.get(session_id,[])
        lc_history = []

        for m in simple:
            role = m.get("role")
            content = m.get("content","")

            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(message, lc_history)

        simple.append({"role":"user","content":message})
        simple.append({"role":"assistant","content":answer})

        SESSION[session_id] = simple

        return ChatResponse(answer=answer)    
    
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=int(os.getenv("PORT","8000")),reload=True)

