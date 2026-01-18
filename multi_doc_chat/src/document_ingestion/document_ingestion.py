from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.utils.config_loader import load_config
import hashlib
import sys

config=load_config()

def generate_session_id() -> str:
    """generate a unique session id with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"

class ChatIngestor():
    def __init__(
            self, temp_base: str = "data",
            faiss_base : str = "faiss_index",
            use_session_dirs : bool = True,
            session_id : Optional[str]=None
        ):
        try:
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialised",
                      session_id=self.session_id,
                      temp_dir= str(self.temp_dir),
                      faiss_dir=str(self.faiss_base),
                      sessionized=self.use_session
            )
        
        except Exception as e:
            log.error("Failed to initialized ChatIngstor",error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor",e) from e



    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base    
    
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documment Split", chunks=len(chunks), chunk_size=chunk_size,overlap = chunk_overlap)
        return chunks
    
    def build_retriever(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = config["retriever"]["k"],
        search_type: str = config["retriever"]["search_type"],
        fetch_k: int = config["retriever"]["fetched_k"],
        lambda_mult: float = config["retriever"]["lambda_mult"],
    ):
        try:
            # 1. Save files
            paths = save_uploaded_files(uploaded_files, self.temp_dir)

            # 2. Load documents
            docs = load_documents(paths)
            if not docs:
                raise DocumentPortalException(
                    "No valid documents could be loaded", None
                )

            # 3. Split documents
            chunks = self._split(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                raise DocumentPortalException(
                    "Document splitting produced zero chunks", None
                )

            # 4. Prepare FAISS inputs
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]

            fm = FaissManager(self.faiss_dir, self.model_loader)

            # 5. Load OR create index (ONLY ONCE)
            vs = fm.load_or_create(
                texts=texts,
                metadatas=metadatas,
            )

            log.info(
                "FAISS index ready",
                index_path=str(self.faiss_dir),
                chunks=len(chunks),
            )

            # 6. Retriever config
            search_kwargs = {"k": k}

            if search_type == "mmr":
                search_kwargs.update(
                    {
                        "fetch_k": fetch_k,
                        "lambda_mult": lambda_mult,
                    }
                )
                log.info(
                    "Using MMR search",
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )

            return vs.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )

        except DocumentPortalException:
            raise

        except Exception as e:
            log.exception("Failed to build retriever")
            raise DocumentPortalException(
                "Failed to build retriever", e
            ) from e
        

class FaissManager:
    def __init__(self, index_dir:Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True,exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str,Any] = {"rows":{}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"row":{}}
            except Exception:
                self._meta = {"rows":{}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embedding()
        self.vs : Optional[FAISS] = None

    def _exists(self)->bool:
        return (self.index_dir / "index.faiss").exists() and \
       (self.index_dir / "index.pkl").exists()

    
    @staticmethod
    def _fingerprint(text:str, md:Dict[str,Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False,indent=2),encoding="utf-8")


    def add_documents(self,docs:List[Document]):
        if self.vs is None:
            raise RuntimeError("call load_or_create() before add_documents_impodent()")

        new_docs : List[Document] = []
        for d in docs:

            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue

            self._meta["rows"][key] = True
            new_docs.append(d) 

            if new_docs:
                self.vs.add_documents(new_docs)
                self.vs.save_local(str(self.index_dir))
                self._save_meta()

            return len(new_docs)

    def load_or_create(self, texts:Optional[List[str]]=None, metadatas: Optional[List[dict]]=None):

        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True
            )
            return self.vs
        
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one",sys)
        self.vs = FAISS.from_texts(texts=texts,embedding=self.emb,metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))

        return self.vs
        
