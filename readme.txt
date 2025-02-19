
# RAG Chat API with FastAPI and LlamaCPP

This project is an API for a chat system using LlamaCPP and Retrieval-Augmented Generation (RAG). It allows users to interact with a language model, providing answers that can be enhanced with information from documents when RAG is enabled.

## Key Features

- **FastAPI**: Used to create a high-performance API.
- **LlamaCPP**: A language model for text generation.
- **RAG (Retrieval-Augmented Generation)**: Enhances responses by retrieving relevant information from documents.
- **Faiss**: Efficient similarity search for document retrieval.
- **HuggingFace Embeddings**: Used for generating text embeddings.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SrMusienko/RAG_for_LLM
   cd rag-chat-api

Install dependencies:

pip install -r requirements.txt

Download the LlamaCPP model and place it in the models/ directory.
 For example, you can use the gemma-2-2b-it.Q8_0.gguf model.
 Web Interface

You can use the web interface available at http://localhost:5000/.
 It allows you to interact with the API through a browser.
## Project Structure

    main.py: The main file containing the API logic.

    static/: Directory for static files, including index.html for the web interface.

    models/: Directory for storing LlamaCPP models.

    data/: Directory for storing documents used in RAG.

##License

This project is licensed under the MIT License. See the LICENSE file for more details.
Author

Sergii Musiienko - sergii.a.musiienko@gmail.com