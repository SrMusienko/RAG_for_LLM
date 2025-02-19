
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import logging
from llama_cpp import Llama
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from functools import lru_cache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Импортируем HuggingFaceEmbedding

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_rag: bool = Field(default=True)

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

@lru_cache(maxsize=1)
def get_llm(model_path: str):
    """Создает и кэширует LlamaCPP LLM."""
    my_model_path = Path(model_path)
    if not my_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        llm = Llama(
            model_path=str(my_model_path),
            model_kwargs={
                "n_ctx": 2048,
                "n_threads": 4,
            },
            temperature=0.7,
            max_new_tokens=512,
        )
        logger.info("LLM created successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise

class RAGChat:
    def __init__(self, llm: Llama, docs_path: str):
        self.llm = llm
        self.chat_history: List[ChatMessage] = []

        try:
            reader = SimpleDirectoryReader(docs_path)
            documents = reader.load_data()

            dimension = 768  # Размерность эмбеддингов для all-MiniLM-L6-v2
            self.faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(self.faiss_index)

            #embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Выбираем модель
            embed_model = HuggingFaceEmbedding(model_name="models/all-MiniLM-L6-v2")
           
           
            self.index = VectorStoreIndex.from_documents(
                documents,
                llm=self.llm,
                vector_store=vector_store,
                embed_model=embed_model  # Указываем модель эмбеддингов
            )

            logger.info(f"Successfully indexed {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    def format_chat_history(self) -> str:
        formatted = "<|begin_of_text|>\n"
        for msg in self.chat_history:
            formatted += f"<|{msg.role}|>\n{msg.content}\n"
        formatted += "<|assistant|>\n"
        return formatted

    def generate_response(self, query: str, use_rag: bool = False) -> str:

        try:
            self.chat_history.append(ChatMessage(role="user", content=query))
            chat_history = self.format_chat_history()
            if use_rag:
                retriever = self.index.as_retriever(
                    similarity_top_k=3,
                )
                nodes = retriever.retrieve(query)

                context = "\n".join([node.text for node in nodes])

                
                prompt = (
                f"Context from documents->:\n{context}<-\n-------------------------\n"
                f"{chat_history}"
                )
            else:
                prompt = chat_history
            context = "No information"
    
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                stop=["<|user|>", "<|end|>"]
            )
            
            model_reply = response["choices"][0]["text"].strip()
            self.chat_history.append(ChatMessage(role="assistant", content=model_reply))
            
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
                
            return model_reply
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Инициализация приложения
app = FastAPI(title="RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализация чата
MODEL_PATH = "models/gemma-2-2b-it.Q8_0.gguf"
llm = get_llm(MODEL_PATH)
rag_chat = RAGChat(llm, docs_path="data/")

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate", response_model=ChatResponse)
async def generate(request: ChatRequest):
    """Генерирует ответ от модели."""
    try:
        response = rag_chat.generate_response(
            request.prompt,
            use_rag=request.use_rag
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(response="", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)