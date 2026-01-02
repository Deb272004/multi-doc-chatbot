import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException

class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY"]

    def __init__(self):
        self.api_keys={}
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.load(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid json object")
                self.api_keys = parsed
                log.info("loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as json",error=str(e))
        

        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        
        missing =  [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys",missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded", keys={k:v[:6] + "..." for k,v in self.api_keys.items()}) 

    def get(self, key:str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val 


class ModelLoader():

    def __init__(self):
        if os.getenv("ENV","local").lower() != "production":
            load_dotenv()
            log.info("Running in local mode: .env loaded")

        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embedding(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("loading embedding model",model=model_name)
            return FastEmbedEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error Loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model",sys)    


    def load_llm(self):
        llm_block = self.config["llm"]      
        provider_key = os.getenv("LLM_PROVIDER","groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config",provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not founf in config")


        llm_config = llm_block[provider_key] 
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature",0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature = temperature,
                max_output_tokens=max_tokens

            )  

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=temperature
            )

        else:
            log.error("Unsupported LLM provider",provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    embedding = loader.load_embedding()
    print(f"Embedding model Loaded {embedding}")
    result = embedding.embed_query("Hello hoe are you?")
    print(f"embedding result : {result}")

    llm = loader.load_llm()
    print(f"LLM loaded: {llm}")
    result = llm.invoke("hello how are you?")
    print(f"LLM result : {result.content}")


      