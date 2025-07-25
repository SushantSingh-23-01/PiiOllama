import gradio as gr
import ollama
import os
from datetime import datetime
import json
from dataclasses import dataclass
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import re
import subprocess

client = chromadb.PersistentClient(path='./chroma_db', settings = Settings(anonymized_telemetry=False))
chroma_collection = client.get_or_create_collection('docs')
print(f'\nChroma DB initialized. \u2705')

@dataclass
class SharedState:
    cwd:str = os.path.dirname(os.path.realpath(__file__))
    chats_dir:str = os.path.join(cwd, 'chats')
    profiles_path:str = os.path.join(cwd, 'profiles.json')
    
    chat_model:str = r'hf.co/mlabonne/gemma-3-4b-it-abliterated-GGUF:Q4_K_M'
    emb_model:str = r'granite-embedding:latest'
    system_prompt:str = r'Act as a helpful assistant'
    temperature:float = 0.9
    top_k:int = 50
    top_p:float = 0.9

    enable_rag:bool = False

def save_chat(history: list, filename: str, shared_state: SharedState) -> tuple[gr.Dropdown, gr.Textbox]:
    if not history:
        return gr.Dropdown(), gr.Textbox()
    
    basename = filename.strip()
    if not basename:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        basename = f'Chat_{time_str}'
    
    if not basename.endswith('json'):
        basename += '.json'
    
    filepath = os.path.join(shared_state.chats_dir, basename)
    with open(filepath, 'w', encoding='utf-8') as f:
        content = [{'role':'system', 'content':shared_state.system_prompt}] + history
        json.dump(content, f, indent=4)
    print(f'\nChat Successfully saved at {filepath} \u2705')
    updated_chats = [basename] + [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return gr.Dropdown(updated_chats, value=basename), gr.Textbox(basename)

def load_chat(filename:str, shared_state:SharedState)->tuple[list, list, SharedState, gr.Textbox]:
    history = []
    if not filename:
        return [], [], shared_state, gr.Textbox()
    filepath = os.path.join(shared_state.chats_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
    print(f'\nLoaded {filename} successfully.\u2705')
    if history[0]['role'] == ['system']:
        shared_state.system_prompt = history[0]['content']
        history = history[1:]
    return history, history, shared_state, gr.Textbox(filename)
    
def delete_chat(filename:str, shared_state:SharedState)->tuple[list, list, gr.Dropdown, gr.Textbox]:
    filepath = os.path.join(shared_state.chats_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    print(f'\nDeleted Chat {filename}. \u2705')
    updated_chats = [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return [], [], gr.Dropdown(updated_chats, value=updated_chats[0]), gr.Textbox('')
    
def simple_text_splitter(text:str, chunk_size:int=256, chunk_overlap:int=64)->list:       
    chunks = []
    if chunk_size > chunk_overlap:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i: i + chunk_size])
    return chunks

def simple_paragraph_splitter(text:str, max_chunk_chars:int, overlap_chars:int):
    paragraphs = re.split(r'\n\n+', text) # split of one or more newlines
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If adding the next paragraph exceeds max_chunk_chars, close current chunk and start new
        # +2 for potential newline
        if len(current_chunk) + len(para) + 2 > max_chunk_chars and current_chunk:
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

def embed_pdfs(files:list, shared_state:SharedState, chunk_size:int, chunk_overlap:int, 
               chunk_method:str)->gr.Textbox:
    if files is None:
        return gr.Textbox('Upload pdf file.')
    char_count = 0
    chunk_count = 0
    for f in files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text().strip()
            char_count += len(text)
            if chunk_method == 'simple':
                chunks = simple_text_splitter(text, chunk_size, chunk_overlap)
            else:
                chunks = simple_paragraph_splitter(text, chunk_size, chunk_overlap)
            
            for j, chunk in enumerate(chunks):
                embeddings = ollama.embed(shared_state.emb_model, chunk)['embeddings']
                unique_ids = f'{os.path.basename(f)}_page{i}_chunk_{j}'
                chroma_collection.add(
                    ids=unique_ids,
                    embeddings=embeddings,
                    documents=chunk
                )
                chunk_count += 1
    print(f'Embedded {chunk_count} chunks. \u2705')            
    return gr.Textbox(
        f'Status: Embedded PDFs in ChromaDB.'
        f'\nTotal Length of Text: {char_count}'
        f'\nTotal chunks count: {chunk_count}'
    )

def ollama_response(message:str, history:list, shared_state:SharedState):
    system_prompt = shared_state.system_prompt 
    if shared_state.enable_rag is True and chroma_collection.count() > 0:
        query_emb = ollama.embed(shared_state.emb_model, message)['embeddings']
        results = chroma_collection.query(
            query_embeddings=query_emb,
            n_results=3,
            include=['documents']
        )
        context = 'Context:\n'
        if results['documents'] is not None:
            for doc in results['documents']:
                context += ' '.join(doc)
        
        system_prompt +='''Based only on the following context, answer the question.\nContext:\n---\n'''
        
        system_prompt += context
        user_message = f'Query: {message}'
        messages = ([{'role':'system', 'content': system_prompt}] + 
                    history + [{'role':'user', 'content':user_message}])
    else:
        messages = ([{'role':'system', 'content': system_prompt}] + history +
                     [{'role':'user', 'content':message}])
    
    history.append({'role':'user', 'content':message}) 
    full_response = ''  
    for chunk in ollama.chat(shared_state.chat_model, messages, stream=True):
        full_response += chunk['message']['content']
        yield ("", 
               history + [{"role": "assistant", "content": full_response}],
               history + [{"role": "assistant", "content": full_response}],
               )
    
    history.append({"role": "assistant", "content": full_response})
    yield gr.Textbox(''), history, history
            
def regen_response(history:list, shared_state:SharedState):
    if len(history) >= 2 and history[-1]['role'] == 'assistant':
        history.pop()
        message = history.pop()['content']
        yield from ollama_response(message, history, shared_state)
    yield '', history, history

def toggle_rag_enable(shared_state:SharedState, toggle:bool)->SharedState:
    shared_state.enable_rag = toggle
    print('RAG Enabled. \u2705')
    return shared_state

def update_settings(shared_state:SharedState, chat_model:str, emb_model:str,
                    system_prompt:str, temperature:float, 
                    top_k:int, top_p:float)->tuple[SharedState, gr.Textbox]:
    shared_state.chat_model = chat_model
    shared_state.emb_model = emb_model
    shared_state.system_prompt = system_prompt
    shared_state.temperature = temperature
    shared_state.top_k = top_k
    shared_state.top_p = top_p
    return shared_state, gr.Textbox('Settings updated')
    
def clear_rag_embeddings()->gr.Textbox:
    all_ids = chroma_collection.get(limit=chroma_collection.count())['ids']
    if all_ids:
        chroma_collection.delete(ids=all_ids)
        print('RAG embeddings cleared. \u2705')
        return gr.Textbox(f"RAG embeddings cleared! Total documents: {chroma_collection.count()} \u2705")
    else:
        return gr.Textbox(f'Chroma collection is already empty.')

def available_ollama_models()->list:
    model_names = []
    models_info = ollama.list()
    for i in models_info['models']:
        model_names.append(i['model'])
    return model_names

def pull_ollama_model(name:str)->tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Textbox]:
    command = ['ollama', 'pull', name]
    try:
        subprocess.run(command, check=True)
        msg = f'Successfully Pulled {name}. \u2705'
    except subprocess.CalledProcessError as e:
        print('Error:', e)
        msg = f'Failed To load Model.'
    except FileNotFoundError:
        msg = f'Ollama not found.'
    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))

def remove_ollama_model(name:str)->tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Textbox]:
    command = ['ollama', 'rm', name]
    try:
        subprocess.run(command, check=True)
        msg = f'Successfully Removed {name}. \u2705'
    except subprocess.CalledProcessError as e:
        print('Error:', e)
        msg = f'Failed To load Model.'
    except FileNotFoundError:
        msg = f'Ollama not found.'
    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))

def load_all_profiles(file_path:str)->dict:
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print(f"{file_path} not Found.\nCreating an Empty File...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            if not isinstance(profiles, dict):
                print(f"Warning: {file_path} content is not a dictionary. Initializing empty profiles.")
                return {}
            return profiles
    except json.JSONDecodeError:
        print(f"Warning: {file_path} is empty/corrupted. Initializing empty profiles.")
        return {}

def save_all_profiles(profiles_data:dict,  shared_state:SharedState)->None:
    file_path = shared_state.profiles_path
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(profiles_data, f, indent=4)
   
def save_profile(shared_state:SharedState, profile_name:str)->tuple[gr.Dropdown, gr.Textbox]:
    all_profiles = load_all_profiles(shared_state.profiles_path)
    if not profile_name.strip():
        return gr.Dropdown(list(all_profiles.keys())), gr.Textbox("Error: Profile name cannot be empty.") 

    data = {
        'chat_model': shared_state.chat_model,
        'system_prompt': shared_state.system_prompt,
        'temperature': shared_state.temperature,
        'top_k': shared_state.top_k,
        'top_p': shared_state.top_p
    }
    all_profiles[profile_name] = data
    save_all_profiles(all_profiles, shared_state)
    print(f'Saved profile {profile_name}. \u2705')
    return (gr.Dropdown(list(all_profiles.keys()), value=profile_name),
            gr.Textbox(f'Profile {profile_name} saved successfully.'))

def load_profile(shared_state:SharedState, profile_name:str)->tuple[SharedState, gr.Textbox]:
    all_profiles = load_all_profiles(shared_state.profiles_path)
    if profile_name in all_profiles:
        shared_state.chat_model = all_profiles[profile_name]['chat_model']
        shared_state.system_prompt = all_profiles[profile_name]['system_prompt']
        shared_state.temperature = all_profiles[profile_name]['temperature']
        shared_state.top_k = all_profiles[profile_name]['top_k']
        shared_state.top_p = all_profiles[profile_name]['top_p']
        msg = f'Loaded Profile {profile_name}.'
        print(msg + ' \u2705')
    else:
        msg = f'Error: Profile {profile_name} not found.'
    return shared_state, gr.Textbox(msg)

def delete_profile(shared_state:SharedState, profile_name:str)->tuple[gr.Dropdown, gr.Textbox, gr.Textbox]:
    if not profile_name or not profile_name.strip(): # Check for None or empty string
        return (gr.Dropdown(choices=list(load_all_profiles(shared_state.profiles_path).keys()), value=None), 
                gr.Textbox(''),
               gr.Textbox("Error: Profile name cannot be empty.", visible=True))

    all_profiles = load_all_profiles(shared_state.profiles_path)
    
    if profile_name in all_profiles:
        del all_profiles[profile_name]

        save_all_profiles(all_profiles, shared_state)
        status_msg = f"Profile '{profile_name}' deleted successfully."
        print(status_msg + ' \u2705')
    else:
        status_msg = f"Error: Profile '{profile_name}' not found. Nothing to delete."

    return (gr.Dropdown(choices=list(all_profiles.keys()), value=None),
            gr.Textbox(''),
           gr.Textbox(status_msg, visible=True))

def get_gpu_memory()-> int:
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    total_memory_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    total_memory_values = [int(x.split()[0]) for _, x in enumerate(total_memory_info)][0]
    return total_memory_values

def get_memory_status(model_name):
    total_gpu_size = get_gpu_memory()/1024
    model_size = 0
    models_info = ollama.list()
    for i in models_info['models']:
        if model_name == i['model']:
            model_size = round(i['size']/(1024)**3, 2)
    size_text = f'{model_size}/{total_gpu_size}'
    if model_size < total_gpu_size:
        status_text = 'No offloading Required.'
    else:
        status_text = 'Model will be partially loaded to RAM'
    return gr.Textbox(f'{size_text} (GBs) | {status_text}')

class ChatUI:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.history = gr.State([])
        
        self._interface()
        self._events()   
    
    def _interface(self):
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
                    choices=[f for f in os.listdir(self.shared_state.value.chats_dir) if f.endswith('json')],
                    label='Chat Files',
                    interactive=True
                )
                self.new_chat_name = gr.Textbox(label='Chat Save Name', interactive=True)
            
            with gr.Row():
                self.load_chat_btn = gr.Button(value='Load Chat', size='md')
                self.save_chat_btn = gr.Button(value='Save Chat', size='md')
                self.delete_chat_btn = gr.Button(value='Delete Chat', size='md')
                
            self.chatbot = gr.Chatbot(
                show_label=False, 
                type="messages",
                height=500,
                )
            
            with gr.Row():
                self.clear_chat_btn = gr.Button('Clear', interactive=True,size='md')
                self.regen_message_btn = gr.Button('Regenerate', interactive=True, size='md')
                self.enable_rag_chkbox = gr.Checkbox(False, label='Enable RAG')
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, 
                                    submit_btn=True, stop_btn=True)
            
        # Documents Page
        gr.Markdown('Docuemnts and RAG')
        with gr.Column(visible=False) as self.docs_page: 
                self.pdf_path_in = gr.File(
                    file_count='multiple',
                    file_types=['.pdf'],
                    label='PDF file',
                )  
                with gr.Row():
                    self.chunk_size = gr.Slider(minimum=128, maximum=1024, value=512, 
                                                step=1, label='Chunk Size', interactive=True)
                    self.chunk_overlap = gr.Slider(minimum=0, maximum=128, value=64, 
                                                step=1, label='Chunk Overlap', interactive=True)
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
            gr.Markdown("### Settings")
            with gr.Accordion('Checkpoints Selection', open=False):
                with gr.Row():
                    self.chat_models_dropdown = gr.Dropdown(
                        available_ollama_models(),
                        label='Choose Chat Model',
                        interactive=True,
                        value=None
                    )
                    self.vram_info = gr.Textbox(label='VRAM Information', scale=0, min_width=350, interactive=False)
                self.emb_models_dropdown = gr.Dropdown(
                    available_ollama_models(),
                    label='Choose Embedding Model',
                    interactive=True,
                    value=None
                )
                
            with gr.Accordion('Model Settings', open=False):
                self.system_prompt_in = gr.Textbox(
                    value='Act as a helpful assistant',
                    label='Chat Model System Prompt',
                    interactive=True, 
                )
                    
                with gr.Row():
                    self.temp_in = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.1,
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
                        step=0.1,
                        value=0.9,
                        label='Top-P',
                        interactive=True
                    )          
                
                with gr.Row():
                    self.profiles_dropdown = gr.Dropdown(
                        choices=list(load_all_profiles(self.shared_state.value.profiles_path)),
                        label='Available Profiles',
                        interactive=True
                    )
                    self.new_profile_name = gr.Textbox(label='New Profile Name', interactive=True)
                    
                with gr.Row():
                    self.save_profile_btn = gr.Button('save Profile')
                    self.load_profile_btn = gr.Button('Load Profile')
                    self.delete_profile_btn = gr.Button('Delete Profile')
            
                
            with gr.Accordion('Download & Remove Models', open=False):
                with gr.Row():
                    self.download_model_name = gr.Textbox(
                        label='Model Name/ HF Directory',
                        placeholder='gemma3:4b',
                        interactive=True, 
                    )
                    self.download_model_btn = gr.Button('Download Model', interactive=True, scale=0, min_width=200)
                with gr.Row():
                    self.delete_model_name = gr.Dropdown(
                        choices=available_ollama_models(),
                        label='Delete Model Name',
                        interactive=True
                    )
                    self.delete_model_btn = gr.Button('Delete model', interactive=True, scale=0, min_width=200)
            
            with gr.Accordion('Miscellaneous',open=False):
                self.chats_dir_input = gr.Textbox(value = self.shared_state.value.chats_dir, 
                                                  label='Chats Directory', interactive=True)
                self.profiles_path_input = gr.Textbox(value=self.shared_state.value.profiles_path,
                                                      label='Profiles', interactive=True)
            self.update_settings_btn = gr.Button('Update Model Settings', size='lg',interactive=True)
            self.settings_status = gr.Textbox(label='Status', interactive=False)

                
    def _events(self):
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
        # chat tab events
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

        # document tab events
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
        
        # settings tab events
        self.chat_models_dropdown.change(
            get_memory_status,
            [self.chat_models_dropdown],
            [self.vram_info]
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
        self.update_settings_btn.click(
            update_settings,
            inputs=[self.shared_state, self.chat_models_dropdown, self.emb_models_dropdown,
                    self.system_prompt_in, self.temp_in, self.top_k_in, self.top_p_in],
            outputs=[self.shared_state, self.settings_status]
        )

with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    ChatUI(app_shared_state)

demo.launch()