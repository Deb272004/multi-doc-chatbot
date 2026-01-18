from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Iterable, List
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exception.custom_exception import DocumentPortalException


SUPPORTED_EXTENSION = {".pdf",".docx",".txt",".pptx",".md",".csv",".xlsx",".xls",".db",".sqlite",".sqlite3"}

log = CustomLogger().get_logger(__name__)

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """save uploaded files (streamlit) and return local paths"""

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved : List[Path] = []
        for uf in uploaded_files:
            name = getattr(uf, "filename",getattr(uf,"name","file"))
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSION:
                log.warning("Unsupported files skipped", filename=name)
                continue

            safe_name = re.sub(r'[^a-zA-Z0-9_\-]','_', Path(name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"

            out= target_dir/fname
            with open(out, "wb") as f:
                if hasattr(uf,"file") and hasattr(uf.file, "read"):
                    f.write(uf.file.read())
                elif hasattr(uf, "read"):
                    data = uf.read()

                    if isinstance(data, memoryview):
                        data = data.tobytes()

                    f.write(data) 

                else:

                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object; no readable interface")
                    
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:    

        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save Uploaded files",e) from e