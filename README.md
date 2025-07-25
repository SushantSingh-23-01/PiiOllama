# PiiOllama

## 💬 Personalized interactive Ollama UI with RAG 

A user-friendly local web interface for interacting with [Ollama](https://ollama.com/) models, featuring Retrieval Augmented Generation (RAG) capabilities using [ChromaDB](https://www.trychroma.com/). This application allows you to chat with various large language models, manage chat history, configure model parameters, and leverage your own PDF documents for enhanced context-aware responses.

## ✨ Features

* **Interactive Chat:** Seamless conversation with Ollama models.
* **Chat History Management:** Save, load, and delete chat sessions.
* **Configurable Models:** Easily select and configure different Ollama chat and embedding models.
* **Custom System Prompts:** Tailor the AI's persona with custom system prompts.
* **Model Parameters:** Adjust temperature, top-k, and top-p for diverse model outputs.
* **Retrieval Augmented Generation (RAG):**
    * Upload PDF documents to create a searchable knowledge base.
    * Automatic text extraction and chunking from PDFs.
    * Embeddings generated using Ollama's embedding models and stored in ChromaDB.
    * Contextual answers derived from your documents when RAG is enabled.
* **Ollama Model Management:** Pull and remove Ollama models directly from the UI.
* **User Profiles:** Save and load predefined model settings as profiles for quick switching.
* **Persistent Data:** Chat histories, profiles, and RAG embeddings are saved locally.

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
**Warning**: `chatui_ollama_rag_proto` (nightly version) has latest updates being made to it. Use `chatui_ollama_rag_stable.py` for smoother experience.
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
   * Toggle "Enable RAG" to utilize uploaded documents for context.

2. **Documents Tab:**
   * Upload PDF Files: Drag and drop your PDF documents here.
   * Chunking Settings: Adjust Chunk Size and Chunk Overlap for how text is broken down. Select 'simple' (character-based) or 'paragraph' chunking.
   * Process & Embed PDFs: Click this button to extract text, chunk it, and embed it into ChromaDB.
   * Clear All RAG Embeddings: Removes all stored document embeddings from ChromaDB.

3. **Settings Tab:**
   * Model Selection: Choose your desired Chat and Embedding models from the dropdowns (must be downloaded first).
   * System Prompt: Customize the initial instructions given to the AI.
   * Model Parameters: Fine-tune Temperature, Top-K, and Top-P for model behavior.
   * User Profiles:
      * Enter a name and click "Save Current Settings as Profile" to store your current model/prompt settings.
      * Select a profile from the dropdown and click "Load Selected Profile" to apply saved settings.
      * "Delete Selected Profile" removes a saved profile.
   * Ollama Model Management:
      * Download Ollama Model: Enter a model name (e.g., mistral:latest, mlabonne/gemma-3-4b-it-abliterated-GGUF:Q4_K_M) and click to download it via Ollama CLI.
      * Delete Ollama Model: Select a locally downloaded model from the dropdown and click to remove it.
      * Apply All Model Settings: Click this button after making changes in the settings tab to apply them to the current session.

## 🐛 Troubleshooting
- **"Ollama server not running" / "Connection refused":** Ensure the Ollama application is running in the background on your system.
- **"Model not found":** Make sure you have pulled the required models using ollama pull <model_name> or via the UI's Settings tab.
- **FileNotFoundError or IOError:** Check file paths and directory permissions. Ensure chats/, chroma_db/, and profiles.json can be created/accessed.
- **PDF Processing Errors:** Ensure your PDFs are not corrupted and are standard PDF format. Large PDFs might take time to process.
- **Gradio UI not loading:** Check your terminal for any Python traceback errors. Ensure all requirements.txt dependencies are installed.

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
