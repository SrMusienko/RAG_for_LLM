from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
from llama_cpp import Llama
import logging
from pathlib import Path

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

class ChatResponse(BaseModel):
    response: str
    error: str = None

class LLMChat:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_threads=4
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

        self.chat_history: List[ChatMessage] = []
        
    def format_chat(self) -> str:
        """Форматирует историю чата для модели."""
        formatted = "<|begin_of_text|>\n"
        for msg in self.chat_history:
            formatted += f"<|{msg.role}|>\n{msg.content}\n"
        formatted += "<|assistant|>\n"
        return formatted

    def generate_response(self, prompt: str) -> str:
        """Генерирует ответ модели с обработкой ошибок."""
        try:
            self.chat_history.append(ChatMessage(role="user", content=prompt))
            context = self.format_chat()
            
            response = self.llm(
                context,
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
app = FastAPI(title="LLM Chat API")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализация чата
MODEL_PATH = "C:\\llama\\models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
chat = LLMChat(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    """Возвращает HTML-страницу чата."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate", response_model=ChatResponse)
async def generate(request: ChatRequest):
    """Генерирует ответ от модели."""
    try:
        response = chat.generate_response(request.prompt)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return ChatResponse(response="", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)