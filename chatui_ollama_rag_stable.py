import gradio as gr
import ollama
import os
from datetime import datetime
import json
from dataclasses import dataclass, field
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import re
import subprocess
import logging
from colorama import Fore, Style, init

# Initialize colorama. This is important for Windows to work correctly.
init()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
    chroma_collection = client.get_or_create_collection('docs')
    logging.info(f"{Fore.BLUE}ChromaDB initialized at: {CHROMA_DB_PATH}{Style.RESET_ALL}")
except Exception as e:
    logging.error(f"{Fore.RED}Failed to initialize ChromaDB: {e}{Style.RESET_ALL}")
    chroma_collection = None 


# --- Data Models ---
@dataclass
class SharedState:
    """Manages the shared state across the Gradio UI."""
    cwd: str = os.path.dirname(os.path.realpath(__file__))
    chats_dir: str = field(default_factory=lambda: os.path.join(SharedState.cwd, 'chats'))
    profiles_path: str = field(default_factory=lambda: os.path.join(SharedState.cwd, 'profiles.json'))
    
    chat_model: str = r'gemma3:4b'
    emb_model: str = r'granite-embedding:latest'
    system_prompt: str = r'Act as a helpful assistant'
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9

    enable_rag: bool = False
    
    def __post_init__(self):
        os.makedirs(self.chats_dir, exist_ok=True)
        logging.info(f"{Fore.BLUE}Chats directory ensured: {self.chats_dir}{Style.RESET_ALL}")
        
        if not os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                logging.info(f"{Fore.GREEN}Created new profiles file: {self.profiles_path}{Style.RESET_ALL}")
            except IOError as e:
                logging.critical(f"{Fore.RED}CRITICAL ERROR: Failed to create profiles file {self.profiles_path} due to permission/IO issue: {e}. Please ensure the application has write access to this directory.{Style.RESET_ALL}")
        else:
            logging.info(f"{Fore.BLUE}Profiles file already exists: {self.profiles_path}{Style.RESET_ALL}")

        logging.info(f"{Fore.BLUE}Profiles Path ensured: {self.profiles_path}{Style.RESET_ALL}")

# --- Utility Functions ---
def parse_loggings(msg:str, color:str) -> str:
    if color == 'green':
        return Fore.GREEN + msg + Style.RESET_ALL
    elif color == 'blue':
        return Fore.BLUE + msg + Style.RESET_ALL
    elif color == 'yellow':
        return Fore.YELLOW + msg + Style.RESET_ALL
    elif color == 'red':
        return Fore.RED + msg + Style.RESET_ALL
    else:
        return Fore.WHITE + msg + Style.RESET_ALL

def load_all_profiles(file_path:str) -> dict:
    """Loads all user profiles from a JSON file."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({}, f) # Create an empty JSON file if it doesn't exist
        logging.warning(parse_loggings(f"{file_path} Doesn't Exsist. Creating an Empty File.", 'yellow'))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            if not isinstance(profiles, dict):
                logging.warning(parse_loggings(f"Content of {file_path} is not a dictionary. Initializing empty profiles.", 'yellow'))
                return {}
            return profiles
    except json.JSONDecodeError as e:
        logging.error(parse_loggings(f"Error decoding JSON from {file_path}: {e}. File might be corrupted.", 'red'))
        return {}
    except IOError as e:
        logging.error(parse_loggings(f"IO Error reading profiles from {file_path}: {e}", 'red'))
        return {}

def save_all_profiles(profiles_data: dict, shared_state: SharedState):
    """Saves all user profiles to a JSON file."""
    file_path = shared_state.profiles_path
    try:
        # Ensure the file itself exists before writing.
        # This handles cases where the file might be deleted after __post_init__ runs.
        if not os.path.exists(file_path): # Added this check
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f) # Create an empty JSON file if it doesn't exist
            logging.info(parse_loggings(f"Re-created missing profiles file before saving: {file_path}", 'blue'))

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=4)
    except IOError as e:
        logging.error(parse_loggings(f"Error saving profiles to {file_path}: {e}", 'red'))
    except Exception as e:
        logging.error(parse_loggings(f"An unexpected error occurred while saving profiles: {e}", 'red')) 

def save_chat(history: list, filename: str, shared_state: SharedState) -> tuple[gr.Dropdown, gr.Textbox]:
    """Saves the current chat history to a JSON file."""
    if not history:
        logging.warning(parse_loggings(f"Attempted to save empty chat history.", 'yellow'))
        return gr.Dropdown(), gr.Textbox()
    
    basename = filename.strip()
    if not basename:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        basename = f'Chat_{time_str}.json'
    elif not basename.endswith('.json'):
        basename += '.json'
    filepath = os.path.join(shared_state.chats_dir, basename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            content = [{'role': 'system', 'content': shared_state.system_prompt}] + history
            json.dump(content, f, indent=4)
        logging.info(parse_loggings(f"Chat saved to: {filepath}", 'green'))
    except IOError as e:
        logging.error(parse_loggings(f"Error saving chat to {filepath}: {e}", 'red'))
        return gr.Dropdown(), gr.Textbox(f"Error saving chat: {e}")
    except Exception as e:
        logging.error(parse_loggings(f"An unexpected error occurred while saving chat: {e}", 'red'))
        return gr.Dropdown(), gr.Textbox(f"Unexpected error saving chat: {e}")
    
    updated_chats = [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return gr.Dropdown(choices=updated_chats, value=basename), gr.Textbox(basename)

def load_chat(filename: str, shared_state: SharedState) -> tuple[list, list, SharedState, gr.Textbox]:
    """Loads a chat history from a JSON file."""
    if not filename:
        logging.info(parse_loggings(f"No chat filename provided for loading.", 'yellow'))
        return [], [], shared_state, gr.Textbox('')
    
    filepath = os.path.join(shared_state.chats_dir, filename)
    history = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        logging.info(parse_loggings(f"Chat loaded from: {filepath}", 'green'))
    except json.JSONDecodeError as e:
        logging.error(parse_loggings(f"Empty or corrupted JSON file: {filepath}. Error: {e}", 'red'))
        return [], [], shared_state, gr.Textbox(f"Error: Corrupted chat file '{filename}'.")
    except IOError as e:
        logging.error(parse_loggings(f"Error loading chat from {filepath}: {e}", 'red'))
        return [], [], shared_state, gr.Textbox(f"Error loading chat: {e}")
    except Exception as e:
        logging.error(parse_loggings(f"An unexpected error occurred while loading chat: {e}", 'red'))
        return [], [], shared_state, gr.Textbox(f"Unexpected error loading chat: {e}")
            
    # Handle system prompt if present in loaded history
    if history and history[0].get('role') == 'system':
        shared_state.system_prompt = history[0]['content']
        history = history[1:]
    return history, history, shared_state, gr.Textbox(filename)
    
def delete_chat(filename: str, shared_state: SharedState) -> tuple[list, list, gr.Dropdown, gr.Textbox]:
    """Deletes a chat file."""
    if not filename:
            return [], [], gr.Dropdown(), gr.Textbox("No chat selected to delete.")
        
    filepath = os.path.join(shared_state.chats_dir, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logging.info(parse_loggings(f"Chat file deleted: {filepath}", 'green'))
        except OSError as e:
            logging.error(parse_loggings(f"Error deleting chat file {filepath}: {e}", 'red'))
            return [], [], gr.Dropdown(), gr.Textbox(f"Error deleting chat: {e}")
    else:
        logging.warning(parse_loggings(f"Attempted to delete non-existent chat file: {filepath}", 'red'))
        
    updated_chats = [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    # Ensure the dropdown value is valid, or set to None if no chats remain
    dropdown_value = updated_chats[0] if updated_chats else None
    return [], [], gr.Dropdown(updated_chats, value=dropdown_value), gr.Textbox('')

def get_chat_files(shared_state_gr: gr.State) -> list[str]:
    """Helper to get chat files from the SharedState's chats_dir."""
    # Access the actual SharedState object via .value
    shared_state = shared_state_gr.value
    if os.path.exists(shared_state.chats_dir):
        return [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return []
    
def simple_text_splitter(text: str, chunk_size: int = 256, chunk_overlap: int = 64) -> list[str]:
    """Splits text into chunks by character count with overlap."""    
    chunks = []
    if chunk_size <= chunk_overlap:
        logging.warning(parse_loggings("Chunk size must be greater than chunk overlap for meaningful splitting.", 'yellow'))
        return [text] # Return original text as a single chunk
    
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i: i + chunk_size])
    return chunks

def simple_paragraph_splitter(text: str, max_chunk_chars: int, overlap_chars: int) -> list[str]:
    """Splits text into chunks based on paragraphs with overlap."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding the next paragraph exceeds max_chunk_chars, close current chunk and start new
        # +2 for potential newline
        if current_chunk and len(current_chunk) + len(para) + 2 > max_chunk_chars:
            chunks.append(current_chunk)
            
            # Create overlap: take last 'overlap_chars' from current_chunk
            current_chunk = current_chunk[-overlap_chars:].strip() if len(current_chunk) >= overlap_chars else ""
            
        if current_chunk:
            current_chunk += "\n\n" + para
        else:
            current_chunk = para

    if current_chunk: # Add the last chunk if any remaining
        chunks.append(current_chunk)
    
    return chunks

def embed_pdfs(files: list, shared_state: SharedState, chunk_size: int, chunk_overlap: int, split_tech: str) -> gr.Textbox:
    """Processes PDF files, extracts text, chunks it, and embeds it into ChromaDB."""
    if files is None:
        return gr.Textbox('upload pdf file')
    if chroma_collection is None:
        return gr.Textbox('Status: ChromaDB is not initialized. Cannot embed.')
    
    char_count = 0
    chunk_count = 0
    
    # Ensure embedding model is available
    try:
        ollama.show(shared_state.emb_model)
    except ollama.ResponseError as e:
        logging.error(parse_loggings(f"Embedding model '{shared_state.emb_model}' not found or accessible: {e}", 'red'))
        return gr.Textbox(f"Error: Embedding model '{shared_state.emb_model}' not found or accessible. Please download it via Settings.")
    
    for f_obj in files:
        try:
            reader = PdfReader(f_obj)
            for i, page in enumerate(reader.pages):
                text = page.extract_text().strip()
                if not text:
                    continue
                
                char_count += len(text)
                if split_tech == 'simple':
                    chunks = simple_text_splitter(text, chunk_size, chunk_overlap)
                else:
                    chunks = simple_paragraph_splitter(text, chunk_size, chunk_overlap)
                
                for j, chunk in enumerate(chunks):
                    if not chunk.strip(): # Skip empty chunks
                        continue
                    unique_id = f'{os.path.basename(f_obj)}_page{i}_chunk_{j}'
                    try:
                        embeddings = ollama.embed(shared_state.emb_model, chunk)['embeddings']
                        chroma_collection.add(
                            ids=unique_id,
                            embeddings=embeddings,
                            documents=chunk
                        )
                        chunk_count += 1
                    except ollama.ResponseError as e:
                            logging.error(parse_loggings(f"Ollama embedding error for chunk: {e}", 'red'))
                            return gr.Textbox(f"Status: Ollama embedding error. Check embedding model and server. Error: {e}")
                    except Exception as e:
                        logging.error(parse_loggings(f"Error embedding chunk {unique_id}: {e}", 'red'))
                        return gr.Textbox(f"Status: Error embedding chunk. Details in logs. Error: {e}")
                
        except Exception as e:
            logging.error(parse_loggings(f"Error processing PDF file {f_obj.name}: {e}", 'red'))
            return gr.Textbox(f"Status: Error processing PDF file '{os.path.basename(f_obj.name)}'. Error: {e}")        
    return gr.Textbox(f'Status: Embedded PDFs in ChromaDB.'
                       f'\nTotal Characters Processed: {char_count}'
                       f'\nTotal Chunks Created: {chunk_count}')

def ollama_response(message: str, history: list, shared_state: SharedState):
    """Generates a response from the Ollama model, optionally with RAG."""
    system_prompt = shared_state.system_prompt 

    # Ensure chat model is available
    try:
        ollama.show(shared_state.chat_model)
    except ollama.ResponseError as e:
        logging.error(parse_loggings(f"Chat model '{shared_state.chat_model}' not found or accessible: {e}", 'red'))
        yield "", history, history
        return gr.Textbox(f"Error: Chat model '{shared_state.chat_model}' not found or accessible. Please download it via Settings.")
    
    if shared_state.enable_rag and chroma_collection and chroma_collection.count() > 0:
        try:
            query_emb = ollama.embed(shared_state.emb_model, message)['embeddings']
            results = chroma_collection.query(
                query_embeddings=query_emb,
                n_results=3,
                include=['documents']
            )
            context = 'Context:\n'
            if results['documents'] is not None:
                for doc_list in results['documents']:
                    context += ' '.join(doc_list) + '\n'
            else:
                logging.warning(parse_loggings(f"No relevant documents found for RAG query.", 'yellow'))
        
            rag_system_prompt = (f"{system_prompt}\n\nBased only on the following context, answer the question." 
                                f"If the answer is not in the context, state that you cannot answer based on the provided information."
                                f"\nContext:\n---\n{context}\n---")
            
            user_message = f'Query: {message}'
            messages = ([{'role':'system', 'content': rag_system_prompt}] + 
                        history + [{'role':'user', 'content':user_message}])
            logging.info(parse_loggings(f"RAG enabled: Querying with context.", 'blue'))
        except ollama.ResponseError as e:
            logging.error(parse_loggings(f"Ollama embedding error during RAG query: {e}", 'red'))
            yield "", history, history
            return gr.Textbox(f"Error: Ollama embedding error during RAG. Check embedding model and server. Error: {e}")
        except Exception as e:
            logging.error(parse_loggings(f"Error during RAG process: {e}", 'red'))
            yield "", history, history
            return gr.Textbox(f"Error: An error occurred during RAG. Details in logs. Error: {e}")

    else:
        messages = ([{'role':'system', 'content': system_prompt}] + history +
                     [{'role':'user', 'content':message}])
        if shared_state.enable_rag and (chroma_collection is None or chroma_collection.count() == 0):
                logging.warning("RAG enabled but ChromaDB is not initialized or empty. Proceeding without RAG.")
    
    
    history.append({'role':'user', 'content':message}) 
    full_response = ''  
    try:
        for chunk in ollama.chat(
            model=shared_state.chat_model, 
            messages=messages, 
            stream=True,
            options={'temperature': shared_state.temperature, 'top_k': shared_state.top_k, 'top_p': shared_state.top_p}
            ):
            full_response += chunk['message']['content']
            yield ("", 
                history + [{"role": "assistant", "content": full_response}],
                history + [{"role": "assistant", "content": full_response}],
                )
    except ollama.ResponseError as e:
        logging.error(f"Ollama chat error: {e}")
        full_response = f"Error: Failed to get response from Ollama. Check model and server. Error: {e}"
        yield "", history + [{"role": "assistant", "content": full_response}], history + [{"role": "assistant", "content": full_response}]
    except Exception as e:
        logging.error(f"An unexpected error occurred during chat generation: {e}")
        full_response = f"Error: An unexpected error occurred. Details in logs. Error: {e}"
        yield "", history + [{"role": "assistant", "content": full_response}], history + [{"role": "assistant", "content": full_response}]
    
    history.append({"role": "assistant", "content": full_response})
    yield '', history, history
            
def regen_response(history: list, shared_state: SharedState):
    """Regenerates the last assistant response."""
    if len(history) >= 2 and history[-1]['role'] == 'assistant':
        history.pop() # Remove last assistant response
        message = history.pop()['content'] # Get last user message
        logging.info(f"{Fore.BLUE}Regenerating response.{Style.RESET_ALL}")
        yield from ollama_response(message, history, shared_state)
    else:
        logging.warning(f"{Fore.YELLOW}Cannot regenerate: Last message is not an assistant response or history is too short.{Style.RESET_ALL}")
        yield "", history, history # Return current state if regeneration is not possible

def toggle_rag_enable(shared_state: SharedState, toggle: bool) -> SharedState:
    """Toggles the RAG (Retrieval Augmented Generation) feature."""
    shared_state.enable_rag = toggle
    logging.info(f"{Fore.BLUE}RAG enabled status: {toggle}{Style.RESET_ALL}")
    return shared_state

def update_settings(shared_state: SharedState, chat_model: str, emb_model: str,
                    system_prompt: str, temperature: float, top_k: int, top_p: float) -> tuple[SharedState, gr.Textbox]:
    """Updates the model settings in the shared state."""
    shared_state.chat_model = chat_model
    shared_state.emb_model = emb_model
    shared_state.system_prompt = system_prompt
    shared_state.temperature = temperature
    shared_state.top_k = top_k
    shared_state.top_p = top_p
    logging.info(f"{Fore.GREEN}Settings updated.{Style.RESET_ALL}")
    return shared_state, gr.Textbox('Settings updated')
    
def clear_rag_embeddings() -> gr.Textbox:
    """Clears all embeddings from the ChromaDB collection."""
    if chroma_collection is None:
        return gr.Textbox('Status: ChromaDB is not initialized.')
    
    current_count = chroma_collection.count()
    if current_count > 0:
        try:
            all_ids = chroma_collection.get(limit=chroma_collection.count())['ids']
            if all_ids:
                chroma_collection.delete(ids=all_ids)
            logging.info(f"{Fore.GREEN}RAG embeddings cleared! Deleted {current_count} documents.{Style.RESET_ALL}")
            return gr.Textbox(f"RAG embeddings cleared! Total documents: {chroma_collection.count()}")
        except Exception as e:
            logging.error(f"{Fore.RED}Error clearing ChromaDB embeddings: {e}{Style.RESET_ALL}")
            return gr.Textbox(f"Error clearing RAG embeddings: {e}")
    else:
        return gr.Textbox(f'Chroma collection is already empty. Total documents: {chroma_collection.count()}')

def available_ollama_models() -> list[str]:
    """Retrieves a list of available Ollama models."""
    model_names = []
    try:
        models_info = ollama.list()
        for model in models_info.get('models', []):
            model_names.append(model['model'])
        logging.info(f"{Fore.GREEN}Successfully fetched available Ollama models.{Style.RESET_ALL}")
    
    # provide user friendly error in dropdown
    except ollama.ResponseError as e:
            logging.error(f"{Fore.RED}Error listing Ollama models: {e}. Is Ollama server running?{Style.RESET_ALL}")
            model_names.append(f"Error: {e}")
    except Exception as e:
        logging.error(f"{Fore.RED}An unexpected error occurred while fetching Ollama models: {e}{Style.RESET_ALL}")
        model_names.append(f"Error: {e}")
    return model_names

def pull_ollama_model(name: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Textbox]:
    """Pulls an Ollama model."""
    if not name.strip():
        return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()),
                gr.Dropdown(available_ollama_models()), gr.Textbox("Model name cannot be empty."))
    command = ['ollama', 'pull', name]
    try:
        logging.info(parse_loggings(f"Attempting to pull Ollama model: {name}", 'blue'))
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        msg = f'Successfully Pulled {name}. Output: {process.stdout}'
        logging.info(parse_loggings(msg, 'green'))
    except subprocess.CalledProcessError as e:
        msg = f'Failed to pull model {name}. Error: {e.stderr}'
        logging.error(parse_loggings(msg, 'red'))
    except FileNotFoundError:
        msg = f'Ollama command not found. Please ensure Ollama is installed and in your PATH.'
        logging.error(parse_loggings(msg, 'red'))
    except Exception as e:
        msg = f'An unexpected error occurred while pulling model: {e}'
        logging.error(parse_loggings(msg, 'red'))

    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))

def remove_ollama_model(name: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Textbox]:
    """Removes an Ollama model."""
    if not name.strip():
        return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()),
                gr.Dropdown(available_ollama_models()), gr.Textbox("Model name cannot be empty."))
    command = ['ollama', 'rm', name]
    try:
        logging.info(parse_loggings(f"Attempting to remove Ollama model: {name}", 'blue'))
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        msg = f'Successfully Removed {name}. Output: {process.stdout}'
        logging.info(parse_loggings(msg, 'green'))   
    except subprocess.CalledProcessError as e:
        msg = f'Failed to remove model {name}. Error: {e.stderr}'
        logging.error(parse_loggings(msg, 'red'))
    except FileNotFoundError:
        msg = f'{Fore.RED}Ollama command not found. Please ensure Ollama is installed and in your PATH.{Style.RESET_ALL}'
        logging.error(parse_loggings(msg, 'red'))
    except Exception as e:
        msg = f'{Fore.RED}An unexpected error occurred while removing model: {e}{Style.RESET_ALL}'
        logging.error(parse_loggings(msg, 'red'))
        
    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))
    
def save_profile(shared_state: SharedState, profile_name: str) -> tuple[gr.Dropdown, gr.Textbox]:
    """Saves the current settings as a user profile."""
    profile_name = profile_name.strip()
    if not profile_name:
        return (gr.Dropdown(list(load_all_profiles(shared_state.profiles_path).keys())),
                gr.Textbox("Error: Profile name cannot be empty.", visible=True))
    
    all_profiles = load_all_profiles(shared_state.profiles_path)

    data = {
        'chat_model': shared_state.chat_model,
        'system_prompt': shared_state.system_prompt,
        'temperature': shared_state.temperature,
        'top_k': shared_state.top_k,
        'top_p': shared_state.top_p
    }
    all_profiles[profile_name] = data
    save_all_profiles(all_profiles, shared_state)
    logging.info(parse_loggings(f"Profile '{profile_name}' saved.", 'green'))
    return (gr.Dropdown(list(all_profiles.keys()), value=profile_name),
            gr.Textbox(f'Profile {profile_name} saved successfully.'))

def load_profile(shared_state: SharedState, profile_name: str) -> tuple[SharedState, gr.Textbox]:
    """Loads settings from a selected user profile."""
    if not profile_name:
        return shared_state, gr.Textbox("Error: No profile selected to load.")

    all_profiles = load_all_profiles(shared_state.profiles_path)

    if profile_name in all_profiles:
        profile_data = all_profiles[profile_name]
        shared_state.chat_model = profile_data.get('chat_model', shared_state.chat_model)
        shared_state.system_prompt = profile_data.get('system_prompt', shared_state.system_prompt)
        shared_state.temperature = profile_data.get('temperature', shared_state.temperature)
        shared_state.top_k = profile_data.get('top_k', shared_state.top_k)
        shared_state.top_p = profile_data.get('top_p', shared_state.top_p)
        msg = f'Loaded Profile "{profile_name}".'
        logging.info(parse_loggings(msg, 'green'))
    else:
        msg = f'Error: Profile {profile_name} not found.'
        logging.error(parse_loggings(msg, 'red'))
    return shared_state, gr.Textbox(msg)

def delete_profile(shared_state: SharedState, profile_name: str) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]:
    """Deletes a user profile."""
    profile_name = profile_name.strip()
    if not profile_name:
        return (gr.Dropdown(choices=list(load_all_profiles(shared_state.profiles_path).keys()), value=None),
                gr.Textbox(''),
                gr.Textbox("Error: Profile name cannot be empty.", visible=True))
        
    all_profiles = load_all_profiles(shared_state.profiles_path)
    
    if profile_name in all_profiles:
        del all_profiles[profile_name]

        save_all_profiles(all_profiles, shared_state)
        status_msg = f"Profile '{profile_name}' deleted successfully."
        logging.info(parse_loggings(status_msg, 'green'))
    else:
        status_msg = f"Error: Profile '{profile_name}' not found. Nothing to delete."
        logging.warning(parse_loggings(status_msg, 'red'))
    
    return (gr.Dropdown(choices=list(all_profiles.keys()), value=None),
            gr.Textbox(''),
           gr.Textbox(status_msg, visible=True))

def get_profile_names(shared_state_gr: gr.State) -> list[str]:
    """Helper to get profile names from the SharedState's profiles_path."""
    # Access the actual SharedState object via .value
    shared_state = shared_state_gr.value
    # Use your existing load_all_profiles function
    profiles = load_all_profiles(shared_state.profiles_path)
    return list(profiles.keys())

def get_total_nvidia_gpu_memory() -> float | None:
    """
    Retrieves the total GPU memory in MiB for the first NVIDIA GPU using nvidia-smi.
    Returns None if nvidia-smi is not available or an error occurs.
    """
    try:
        # Using -q -d memory.total -o default for cleaner output on Windows
        command = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(command, encoding='utf-8').strip()
        # Expecting output like "X MiB" or "X" if no units.
        # Splitting by newline and taking the first line for the first GPU
        memory_str = output.split('\n')[0].strip()
        return int(memory_str) / 1024 # nvidia-smi usually reports in MiB by default for memory.total
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.error(parse_loggings(f"Failed to get GPU memory: {e}", 'red'))
        logging.info(parse_loggings("Please ensure NVIDIA drivers are installed and nvidia-smi is in your PATH.", 'blue'))
        return None

def get_ollama_model_size_gb(model_name: str) -> float | None:
    """
    Fetches the size of a specific Ollama model in GB.
    Returns None if the model is not found or an error occurs.
    """
    try:
        models_info = ollama.list()
        for model_data in models_info.get('models', []): # Use .get with default empty list
            if model_data.get('model') == model_name:
                size_bytes = model_data.get('size')
                if size_bytes is not None:
                    return round(size_bytes / (1024)**3, 2)
                else:
                    logging.warning(parse_loggings(f"Model '{model_name}' found, but 'size' information is missing.", 'yellow'))
                    return None
        logging.warning(parse_loggings(f"Model '{model_name}' not found in Ollama list.", 'yellow'))
        return None
    except Exception as e:
        logging.error(parse_loggings(f"Error fetching Ollama model list: {e}", 'red'))
        return None

def get_memory_status(model_name: str) -> gr.Textbox:
    """
    Checks model's VRAM requirement against total GPU memory and provides status.
    Handles various error scenarios gracefully for Gradio display.
    """
    total_gpu_size_gb = get_total_nvidia_gpu_memory()

    if total_gpu_size_gb is None:
        return gr.Textbox("Error: Could not determine GPU memory. Is `nvidia-smi` installed and in PATH?\u2757",
                           label="Status", interactive=False, visible=True,
                           elem_classes="error-text") # Add CSS class for red text

    model_size_gb = get_ollama_model_size_gb(model_name)

    if model_size_gb is None:
        return gr.Textbox(f"Error: Could not determine size for model '{model_name}'. Is Ollama running and model downloaded?\u2757",
                           label="Status", interactive=False, visible=True,
                           elem_classes="error-text")

    status_message = ""
    size_display = f"{model_size_gb:.2f} GB / {total_gpu_size_gb:.2f} GB (GPU Total)"

    if model_size_gb <= total_gpu_size_gb:
        status_message = "No offloading required. Model fits in VRAM.\u2705"
        #color_class = "green-text"
    else:
        offload_amount_gb = model_size_gb - total_gpu_size_gb
        status_message = f"Offloading likely required. Model exceeds VRAM by {offload_amount_gb:.2f} GB.\u274c"
        #color_class = "red-text"
        
    return gr.Textbox(f"Model Size: {size_display}\nStatus: {status_message}",
                       label="VRAM Compatibility", interactive=False, visible=True)
    
# --- Gradio UI Class ---

class ChatUI:
    def __init__(self, shared_state: gr.State):
        self.shared_state = shared_state
        self.history = gr.State([])
        
        self._interface()
        self._events()   
    
    def _interface(self):
        """Defines the Gradio UI layout."""
        with gr.Sidebar():
            gr.Markdown("## Navigation")
            self.chat_tab_btn = gr.Button("Chat")
            self.docs_tab_btn = gr.Button('Documents')
            self.settings_tab_btn = gr.Button("Settings")
            
        # Chat Page
        with gr.Column(visible=True) as self.chat_page:
            gr.Markdown("### Chat Interface")
            with gr.Row():
                self.chats_dropdown = gr.Dropdown(
                    choices=get_chat_files(self.shared_state),
                    label='Chat Files',
                    interactive=True,
                    value=None
                )
                self.new_chat_name = gr.Textbox(label='Chat Save Name', interactive=True, 
                                                placeholder="Enter name to save/delete chat")
            
            with gr.Row():
                self.load_chat_btn = gr.Button(value='Load Chat', size='md')
                self.save_chat_btn = gr.Button(value='Save Chat', size='md')
                self.delete_chat_btn = gr.Button(value='Delete Chat', size='md')
                
            self.chatbot = gr.Chatbot(
                show_label=False, 
                type="messages",
                height=500,
                value=[]
                )
            
            with gr.Row():
                self.clear_chat_btn = gr.Button('Clear', interactive=True, size='md')
                self.regen_message_btn = gr.Button('Regenerate', interactive=True, size='md')
                self.enable_rag_chkbox = gr.Checkbox(False, label='Enable RAG')
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, 
                                    submit_btn=True, placeholder="Type your message here...")
            
        # Documents Page
        with gr.Column(visible=False) as self.docs_page: 
            gr.Markdown("### Document Processing for RAG")
            self.pdf_path_in = gr.File(
                file_count='multiple',
                file_types=['.pdf'],
                label='PDF file',
            )  
            with gr.Row():
                self.chunk_size = gr.Slider(minimum=128, maximum=2048, value=512, 
                                            step=1, label='Chunk Size (Characters)', interactive=True)
                self.chunk_overlap = gr.Slider(minimum=0, maximum=512, value=64, 
                                            step=1, label='Chunk Overlap (Characters)', interactive=True)
                self.chunking_method = gr.Dropdown(
                    choices=['simple', 'paragraph'],
                    value='paragraph',
                    label='Chunking Method',
                    interactive=True
                )
            with gr.Row():
                self.process_pdf_btn = gr.Button('Process PDF', interactive=True)
                self.clear_embds_btn = gr.Button('Delete Embeddings', interactive=True)
            self.pdf_status = gr.Textbox(label='pdf status', interactive=False, lines=3)
                
        # Settings Page
        with gr.Column(visible=False) as self.settings_page:
            gr.Markdown("### Application Settings")
            with gr.Accordion('Model Selection & Paths', open=False):
                with gr.Row():
                    self.chat_models_dropdown = gr.Dropdown(
                        available_ollama_models(),
                        label='Choose Chat Model',
                        interactive=True,
                        value=None,
                    )
                    self.vram_info_dis = gr.Textbox(label='VRAM Information', interactive=False, lines=2)
                self.emb_models_dropdown = gr.Dropdown(
                    available_ollama_models(),
                    label='Choose Embedding Model',
                    interactive=True,
                    value=None,
                )

                self.chats_dir_input = gr.Textbox(value = 'chats', label='Chats Directory', interactive=True)
                self.profiles_path_input = gr.Textbox(value='profiles.json',label='Profiles', interactive=True)
                
            with gr.Accordion('Chat Model Parameters & Profiles', open=False):
                self.system_prompt_in = gr.Textbox(
                    value='Act as a helpful assistant',
                    label='Chat Model System Prompt',
                    interactive=True, 
                    lines=3
                )
                    
                with gr.Row():
                    self.temp_in = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.05,
                        value=0.9,
                        label='Temperature',
                        interactive=True
                    )
                    self.top_k_in = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=40,
                        label='Top-K',
                        interactive=True
                    )
                    self.top_p_in = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.9,
                        label='Top-P',
                        interactive=True
                    )          
                
                with gr.Row():
                    self.profiles_dropdown = gr.Dropdown(
                        choices=get_profile_names(self.shared_state),
                        label='Available Profiles',
                        interactive=True
                    )
                    self.new_profile_name = gr.Textbox(label='New Profile Name', interactive=True,
                                                       placeholder="Enter name to save current settings")
                    
                with gr.Row():
                    self.save_profile_btn = gr.Button('save Profile')
                    self.load_profile_btn = gr.Button('Load Profile')
                    self.delete_profile_btn = gr.Button('Delete Profile')
            
                
            with gr.Accordion('Ollama Model Management', open=False):
                gr.Markdown("Manage Ollama models (pull from HuggingFace/Ollama registry or delete locally).")
                with gr.Row():
                    self.download_model_name = gr.Textbox(
                        label='Checkpoint Name/ HF Directory',
                        placeholder='Enter full model name or path',
                        interactive=True, 
                    )
                    self.download_model_btn = gr.Button('Download Model', interactive=True)
                with gr.Row():
                    self.delete_model_name = gr.Dropdown(
                        choices=available_ollama_models(),
                        label='Name of checkpoint to delete',
                        interactive=True,
                        value=None
                    )
                    self.delete_model_btn = gr.Button('Delete Model', interactive=True)
            self.update_settings_btn = gr.Button('Apply All Model Settings', size='lg', interactive=True)
            self.settings_status = gr.Textbox(label='Settings Status', interactive=False, lines=2)
                
    def _events(self):
        """Defines the event handlers for UI components."""
        self.chat_tab_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.docs_tab_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.settings_tab_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        # Chat Page Events
        self.msg.submit(
            fn=ollama_response,
            inputs=[self.msg, self.history, self.shared_state],
            outputs= [self.msg, self.history, self.chatbot],
        )
        self.save_chat_btn.click(
            save_chat,
            [self.history, self.new_chat_name, self.shared_state],
            [self.chats_dropdown, self.new_chat_name]
        )
        self.load_chat_btn.click(
            load_chat,
            [self.chats_dropdown, self.shared_state],
            [self.history, self.chatbot, self.shared_state, self.new_chat_name]
        )
        self.delete_chat_btn.click(
            delete_chat,
            [self.new_chat_name, self.shared_state],
            [self.history, self.chatbot, self.chats_dropdown, self.new_chat_name]
        )
        self.clear_chat_btn.click(
            lambda: ([], []),
            outputs=[self.history, self.chatbot]
        )
        self.regen_message_btn.click(
            regen_response,
            [self.history, self.shared_state],
            [self.msg, self.history, self.chatbot]
        )

        self.enable_rag_chkbox.change(
            toggle_rag_enable,
            [self.shared_state, self.enable_rag_chkbox],
            [self.shared_state]
        )
        
        # Document Page Events
        self.process_pdf_btn.click(
            fn=embed_pdfs,
            inputs=[self.pdf_path_in, self.shared_state, self.chunk_size, 
                    self.chunk_overlap, self.chunking_method],
            outputs=[self.pdf_status]
        )
        self.clear_embds_btn.click(
            clear_rag_embeddings,
            outputs=[self.pdf_status]
        )
        
        # Settings Page Events
        self.chat_models_dropdown.change(
            get_memory_status,
            [self.chat_models_dropdown],
            [self.vram_info_dis]
        )
        self.update_settings_btn.click(
            update_settings,
            inputs=[self.shared_state, self.chat_models_dropdown, self.emb_models_dropdown,
                    self.system_prompt_in, self.temp_in, self.top_k_in, self.top_p_in],
            outputs=[self.shared_state, self.settings_status]
        )
        
        self.download_model_btn.click(
            pull_ollama_model,
            [self.download_model_name],
            [self.chat_models_dropdown, self.emb_models_dropdown, 
             self.delete_model_name, self.settings_status]
        )
        self.delete_model_btn.click(
            remove_ollama_model,
            [self.delete_model_name],
            [self.chat_models_dropdown, self.emb_models_dropdown, 
             self.delete_model_name, self.settings_status]
        )
        self.load_profile_btn.click(
            load_profile,
            [self.shared_state, self.profiles_dropdown],
            [self.shared_state, self.settings_status]
        )
        self.save_profile_btn.click(
            save_profile,
            [self.shared_state, self.new_profile_name],
            [self.profiles_dropdown, self.settings_status]
        )
        self.delete_profile_btn.click(
            delete_profile,
            [self.shared_state, self.new_profile_name],
            [self.profiles_dropdown, self.new_profile_name, self.settings_status]
        )

with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    ChatUI(app_shared_state)

demo.launch()
