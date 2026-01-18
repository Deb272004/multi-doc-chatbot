from __future__ import annotations

import os
import sys
from operator import itemgetter
from typing import Optional, Dict, Any, List

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.schema.schema import PromptType, ChatAnswer
from multi_doc_chat.utils.config_loader import load_config
from langchain_core.prompts import MessagesPlaceholder

config = load_config()


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss("faiss_index/abc")
        answer = rag.invoke("What is RAG?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id
            self.llm = self._load_llm()

            # ---- FIX 1: Correct prompt key spelling ----
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALISED_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever = retriever
            self.chain = None

            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException(
                "Initialization error in ConversationalRAG", e
            ) from e

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = config["retriever"]["k"],
        search_type: str = config["retriever"]["search_type"],
        fetched_k: int = config["retriever"]["fetched_k"],
        index_name: str = "index",
        lambda_mult: float = config["retriever"]["lambda_mult"],
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            if search_type not in {"similarity", "mmr"}:
                raise ValueError(f"Unsupported search_type: {search_type}")

            embeddings = ModelLoader().load_embedding()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs.update(
                        {"fetch_k": fetched_k, "lambda_mult": lambda_mult}
                    )

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded",
                index_path=index_path,
                search_type=search_type,
                k=k,
                session_id=self.session_id,
            )

            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException(
                "Loading error of retriever in ConversationalRAG", e
            ) from e

 
    def invoke(
        self, user_input: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> str:
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() first.",
                    None,
                )

            chat_history = chat_history or []

            payload = {
                "input": user_input,
                "chat_history": chat_history,
            }

            answer = self.chain.invoke(payload)

            if not answer:
                log.warning("No answer generated", session_id=self.session_id)
                return "No answer generated."

            # ---- FIX: schema validation ----
            validated = ChatAnswer(answer=str(answer))
            return validated.answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException(
                "Invocation error in ConversationalRAG", e
            ) from e

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")

            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm   # ✅ CRITICAL FIX

        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException(
                "LLM loading error in ConversationalRAG", e
            ) from e

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(
            getattr(d, "page_content", str(d)) for d in docs
        )

    @staticmethod
    def _format_chat_history(history: List[BaseMessage]) -> str:
        return "\n".join(msg.content for msg in history)

    

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise ValueError("Retriever must be set before building LCEL chain")
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieved_docs = (
                question_rewriter
                | self.retriever
                | ConversationalRAG._format_docs
            )

            self.chain = (
                {
                    "context": retrieved_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL chain built successfully", session_id=self.session_id)

        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e))
            raise DocumentPortalException("Failed to build LCEL chain", e) from e
