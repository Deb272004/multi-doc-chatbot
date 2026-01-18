from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from typing import Iterable, List
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,CSVLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

SUPPORTED_EXTENSION = {".pdf",".docx",".txt",".csv"}

def load_documents(paths: Iterable[Path]) -> List[Document]:
    """load docs using appropiate loader based on extension"""

    docs : List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTENSION:
                log.warning("Unsupported extension skipped",path=str(p))
                continue
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p))
            elif ext == ".csv":
                loader = CSVLoader(str(p))
            else:
                log.warning("Unsupported extnsion skipped",path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents",e) from e
    


class FastAPIFileAdapter:
    """
    Adapt FastAPI uploadfile to a simple object with .name and .getbuffer().
    """

    def __init__(self, uf:UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) ->bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()