import os 
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.document_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG

load_dotenv()

def test_document_ingestion_and_rag():
    try:
        test_files = [
            "/home/deblina/Documents/multi-doc-chatbot/data/agenticAI.txt"
        ]
        uploaded_files = []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path,"rb"))
            else:
                print(f"file doesnot exist:{file_path}")
            
            if not uploaded_files:
                print("no valid files to upload")
                sys.exit(1)
            
            chat_ingestor = ChatIngestor(use_session_dirs=True)
            retriever = chat_ingestor.build_retriever(uploaded_files)

            for f in uploaded_files:
                try:
                    f.close()
                except Exception:
                    pass

            session_id = chat_ingestor.session_id
            index_dir = os.path.join("faiss_index",session_id)

            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_faiss(index_path=index_dir,index_name=os.getenv("FAISS_INDEX_NAME","index"))

            chat_history = []
            
            print("\n Type 'exit' to quit the chat")

            while True:
                try:
                    user_input = input("you: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n Exiting chat")
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"exit","quit","q",":q"}:
                    print("goodbye!")
                    break

                answer = rag.invoke(user_input,chat_history=chat_history)
                print("Assistant:",answer)

    except Exception as e:
        print(f"Test failed",{str(e)})
        sys.exit(1)

if __name__ == "__main__":
    test_document_ingestion_and_rag()

