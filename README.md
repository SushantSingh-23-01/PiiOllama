# PiiOllama

## 💬 Personalized interactive Ollama UI with RAG 

A user-friendly local web interface for interacting with [Ollama](https://ollama.com/) models, featuring Retrieval Augmented Generation (RAG) capabilities using [ChromaDB](https://www.trychroma.com/). This application allows you to chat with various large language models, perform Retrieval-Augmented Generation (RAG) from PDF documents, real-time web search results from DuckDuckGo and summarize documents.


## ✨ Features

-   **Local LLM Integration**: Connects seamlessly with any model served through Ollama.
-   **Retrieval-Augmented Generation (RAG)**: Ingest PDF documents to build a local knowledge base that the LLM can use to answer questions.
-   **Live Web Search**: Augment conversations with up-to-date information from the internet using the DuckDuckGo search API.
-   **MapReduce Summarization**: Summarize long documents efficiently by breaking them into chunks, summarizing each individually, and then combining the results.
-   **Full Chat Management**: Save and load chat histories to resume conversations at any time.
-   **Configurable Settings**: A detailed settings panel to control model parameters, RAG configurations, and directory paths.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
1.  **Ollama:** Download and install Ollama from [ollama.com](https://ollama.com/). Make sure the Ollama server is running in the background. Pull a Chat Model and an Embedding Model
    * **Verify Installation:** Open your terminal and run `ollama list`. You should see output similar to:
      ```
      NAME                         ID              SIZE      MODIFIED
      gemma3:4b                    ...              ...      ...
      granite-embedding:latest     ...              ...      ...                             
      ```
2.  **Python:** Python 3.11+ is recommended.
    * **Verify Installation:** `python --version`

3.  Download Ollama Models

This application defaults to `gemma3:4b` for chat and `granite-embedding:latest` for embeddings (used by RAG). You can download these models using the Ollama CLI:

```bash
ollama pull gemma3:4b
ollama pull granite-embedding:latest
```

### 🛠️ Environment Setup

1. Clone the Repository
```
https://github.com/SushantSingh-23-01/PiiOllama.git
cd PiiOllama
```

2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

3. Run the Application

**Warning**: `chatui_ollama_rag_proto.py` (nightly version) has latest updates being made to it. Use `chatui_ollama_rag_stable.py` for smoother experience.
```bash
python chatui_ollama_rag_stable.py
```

## 📂 Project Structure
```
ollama-rag-chat-ui/
├── chatui_rag_production.py  # Main application code
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Files/directories to ignore in Git
├── chats/                    # Directory for saved chat histories (created on first run)
└── chroma_db/                # Directory for ChromaDB embeddings (created on first run)
└── profiles.json             # Stores saved user profiles (created on first run)
```

## ⚙️ Configuration & Data Persistence
- **Chat Histories**: Saved chats are stored as JSON files in the chats/ directory.
- **User Profiles**: Model settings saved as profiles are stored in profiles.json.
- **RAG Embeddings**: The vector database (ChromaDB) for RAG is persistently stored in the chroma_db/ directory.
- **Default Model Configuration**: Default models and parameters are set within chatui_rag_production.py. These can be overridden in the UI's "Settings" tab.

## 🌐 How to Use the UI
1. **Chat Tab:**
   * Type your messages in the input box at the bottom.
   * Use "Save Chat" to store your conversation.
   * "Load Chat" and "Delete Chat" manage saved sessions.
   * "Clear Chat" empties the current conversation.
   * "Regenerate Last Response" re-prompts the model for the previous user message.
   * Toggle "RAG" to utilize uploaded documents for context.
   * Toggle "DuckDuckGo Search" for upto date web search context.

2. **Documents Tab:**
   * **Reading**: Read PDF into memory.
   * **RAG**: Ingesting text chunks and embeddings in vectorstores.
   * **Summarization**: Summarize and save document text as markdown. 

3. **Settings Tab:**
   * Model Selection: Choose your desired Chat and Embedding models from the dropdowns (must be downloaded first).
   * System Prompt: Customize the initial instructions given to the AI.
   * Model Parameters: Fine-tune Temperature, Top-K, and Top-P for model behavior.
   * Character Names: For more consistent chat flow, give names to user and assitant.
   * Additional Context & Directory settings: Document retrieval parameters and default directories can be modified here.

## 🐛 Troubleshooting
- Most of the **ERROR** logs are displayed in terminal for more information.
- **TO DOs** Implement error and warning logs to displayed in ui itself. 

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
