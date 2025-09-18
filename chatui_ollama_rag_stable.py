import gradio as gr
import os 
import json
from datetime import datetime
import ollama
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import re
import unicodedata
import hashlib
import pymupdf
import time
from ddgs import DDGS


class SharedState:
    def __init__(self) -> None:
        self.chat_model = 'hf.co/mlabonne/gemma-3-4b-it-abliterated-GGUF:Q4_K_M'
        self.embed_model = 'snowflake-arctic-embed:33m'
        self.temperature = 1.0
        self.top_k = 64
        self.top_p = 0.95
        self.num_ctx = 2048
        
        self.user_name = 'User'
        self.bot_name = 'Assistant'
        self.system_prompt = 'You are a helpful assistant.'
        
        self.chats_dir = 'chats'
        
        self.documents_text = ''
        self.chromadb_dir = ''
        self.chunk_size = 450
        self.chunk_overlap = 45
        self.n_results = 5
        self.rag_flag = False
        self.ddgs_flag = False
    
        self.sum_dir = 'summaries'
        self.sum_text = ''
        

    def _update_model_settings(
        self,
        chat_model,
        emb_model,
        system_prompt,
        temperature,
        top_k, 
        top_p,
        num_ctx,
        user_name,
        assistant_name
    ):

        self.chat_model = chat_model
        self.embed_model = emb_model

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_ctx = int(2**(num_ctx))
        self.user_name = user_name if user_name else 'N.A.'
        self.assistant_name = assistant_name if assistant_name else 'N.A.'
    
        
        status_dict = {
            'Chat Model': self.chat_model,
            'Embedding Model': self.embed_model,
            'System Prompt': self.system_prompt,
            'Temperature': self.temperature,
            'Top-K': self.top_k,
            'Top-P': self.top_p,
            'Context Length': self.num_ctx,
            'user_name': self.user_name,
            'assistant_name': self.assistant_name
        }
        
        max_key_length = max(len(k) for k in status_dict.keys())
        status_output = 'Updated Settings:\n' + '-'*100 + '\n'
        for k, v in status_dict.items():
            status_output += f'{k:<{max_key_length+10}} {v}\n'
        print('INFO: Updated Model Settings.')
        return self, f'```\n{status_output}\n```'

    def _update_rag_settings(self, n_results, chromadb_dir):
        self.n_results = n_results
        self.chromadb_dir = chromadb_dir
        status_dict = {
            'n_results': self.n_results,
            'chromadb_dir': self.chromadb_dir
        }
        max_key_length = max(len(k) for k in status_dict.keys())
        status_output = 'Updated Settings:\n' + '-'*100 + '\n'
        for k, v in status_dict.items():
            status_output += f'{k:<{max_key_length+10}} {v}\n'
        print('INFO: Updated Model Settings.')
        return self, f'```\n{status_output}\n```'
    
    def _toggle_rag_flag(self, rag_flag_in):
        self.rag_flag = rag_flag_in
        if self.rag_flag is True:
            print(f'INFO: RAG Enabled')
        return self

    def _toggle_ddgs_flag(self, ddgs_flag_in):
        self.ddgs_flag = ddgs_flag_in
        if self.ddgs_flag is True:
            print(f'INFO: Duckduckgo search Enabled')
        return self
    
class FileManager:
    def _load_chat(self, filename, state):
        history = []
        
        # ensure chats folder exsists
        os.makedirs(state.chats_dir, exist_ok=True)
        filepath = os.path.join(state.chats_dir, filename.strip())
        
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if history[0].get('role', []) == 'system':
            state.sp = history[0]['content']
            history = history[1:]
        print(f'INFO: Loaded chat {filename}', 'i')
        return history, history, state, gr.Textbox(filename)
    
    def _save_chat(self, filename, history, state):
        '''
        Save chat files in JSON format.
        '''
        if not history:
            print('WARNING: Trying to save empty chat history.', 'w')
            return gr.Dropdown(), gr.Textbox()
        
        # remove trailing whitespaces 
        basename = filename.strip()
        
        # create a unique name if new chat name is not provided
        if not basename:
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
            basename = f'Chat_{time_str}'
        
        # ensure the extension in json
        if not basename.endswith('json'):
            basename += '.json'
        
        # ensure chats folder exsists
        os.makedirs(state.chats_dir, exist_ok=True)
        filepath = os.path.join(state.chats_dir, basename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # prepend system prompt to history
            content = [{'role': 'system', 'content': state.sp}] + history
            json.dump(content, f, indent=4)
        print(f'INFO: Saved Chat successfully at: {filepath}')
        
        update_chats = [f for f in os.listdir(state.chats_dir) if f.endswith('json')]
        return gr.Dropdown(update_chats, value=basename),  gr.Textbox(basename)
    
def _get_ollama_models():
    model_names = []
    models_info = ollama.list()    
    for i in models_info['models']:
        model_names.append(i['model'])
    return model_names  

class PdfReader:
    def clean_text(self, text):
        '''Light cleaning method for PDF texts'''
        # Remove citations like [1], [1, 2], etc.
        text = re.sub(r'\[\s*\d+(?:,\s*\d+)*\s*\]', '', text)
        # Replace curly quotes with straight quotes
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)
        # Handle hyphenated words at line breaks
        text = re.sub(r'([a-zA-Z])-\s*\n', r'\1', text)
        #  Replace multiple newlines with a single space
        text = re.sub(r'\n+', ' ', text)
        # Normalize whitespace and strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        return text

    def _read_pdf(self, filename):
        total_text = ''
        page_count = 0
        try:
            doc = pymupdf.open(filename)
            page_count = doc.page_count
            for page in doc:
                text = page.get_text()
                total_text += self.clean_text(text)
            doc.close()
        except Exception as e:
            print(f'ERROR: Error ocurred reading PDF. {e}')
        return total_text, page_count
    
    def _read_pdfs(self, filenames, state):
        print('INFO: Reading PDFs...')
        start = time.time()
        total_text = ''    
        file_details = []
        

        for f in tqdm(filenames):
            file_text, page_count = self._read_pdf(f)

            total_text += file_text
            
            # prepare metadata for indivual files
            file_size_bytes = os.path.getsize(f)
            file_details.append({
                'filename': os.path.basename(f),
                'text_length': len(file_text),
                'page_count': page_count,
                'PDF size (MB)': round(file_size_bytes / (1024)**2, 2),
            })
            
        end = time.time()
        print('INFO: Finished Reading PDFs.')
        total_time = end - start
        metadata = {
            'number_of_pdfs': len(filenames),
            'total_text_length': len(total_text),
            'total_time_seconds': round(total_time, 2),
            'file_details': file_details
        }
        
        max_key_length = max(len(k) for k in metadata.keys())
        metadata_text = ''
        for k, v in metadata.items():
            metadata_text += f'{k:<{max_key_length+10}} {v}\n'
        
        state.documents_text = total_text
        return state, '**Summary**:\n\n' + f'```{metadata_text}```'
    
class RecursiveSplitter:
    def split_text(
        self, 
        text, 
        chunk_size,
        chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
        ):
        """
        The main public method to split text into a list of chunks.
        """
        return self._recursive_split(text, chunk_size, chunk_overlap, separators)

    def _recursive_split(
        self, 
        text, 
        chunk_size,
        chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
        ):
        """The core recursive splitting logic."""
        final_chunks = []
        
        # If the text is empty or None, return an empty list
        if not text:
            return []

        # If the text is already small enough, return it as a single chunk.
        if len(text) <= chunk_size:
            return [text]

        # Get the highest-priority separator for this level of recursion.
        current_separator = separators[0]
        next_separators = separators[1:]
        
        # If we've reached the end of our separators list (the empty string),
        # we perform a simple character-level split as a last resort.
        if current_separator == "":
            return self._character_split(text, chunk_size, chunk_overlap)

        # Split the text by the current separator. The regex uses a capture group `()`
        # which keeps the separator in the resulting list, allowing us to preserve it.
        # e.g., re.split('(sep)', 'text1septext2') -> ['text1', 'sep', 'text2']
        try:
            splits = re.split(f'({re.escape(current_separator)})', text)
        except re.error as e:
            # If the separator is invalid for regex, skip it and recurse.
            print(f"ERROR: Invalid regex separator '{current_separator}'. Skipping and recursing. Error: {e}")
            return self._recursive_split(text, chunk_size, chunk_overlap, next_separators)

        # Merge the text parts with their subsequent separators.
        # This turns ['text1', 'sep', 'text2'] into ['text1sep', 'text2'].
        merged_splits = []
        for i in range(0, len(splits), 2):
            chunk = splits[i]
            # If there's a separator that follows this chunk, append it.
            if i + 1 < len(splits):
                chunk += splits[i+1]
            if chunk:
                merged_splits.append(chunk)

        # Now, process these merged splits: merge small ones, and recursively split any that are still too large.
        current_chunk_accumulator = ""
        for split in merged_splits:
            # If a single split is *still* larger than our chunk size, we must
            # recursively split it further using the *next* set of separators.
            if len(split) > chunk_size:
                if current_chunk_accumulator:
                    final_chunks.append(current_chunk_accumulator)
                    current_chunk_accumulator = ""
                
                sub_chunks = self._recursive_split(split, chunk_size, chunk_overlap, next_separators)
                final_chunks.extend(sub_chunks)
            
            # If adding the next split would make our accumulated chunk too large,
            # we finalize the current accumulator and start a new one.
            elif len(current_chunk_accumulator) + len(split) > chunk_size:
                if current_chunk_accumulator:
                    final_chunks.append(current_chunk_accumulator)
                current_chunk_accumulator = split

            # Otherwise, the split fits, so we add it to the accumulator.
            else:
                current_chunk_accumulator += split
        
        # Add the last remaining accumulated chunk to our results.
        if current_chunk_accumulator:
            final_chunks.append(current_chunk_accumulator)
            
        # Filter out any chunks that are only whitespace.
        return [chunk for chunk in final_chunks if chunk.strip()]

    def _character_split(self, text, chunk_size, chunk_overlap):
        chunks = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = start_idx + chunk_size
            chunks.append(text[start_idx:end_idx])
            start_idx += chunk_size - chunk_overlap
        return chunks

class ParentChildsplitter:
    def __init__(
        self,
        chromdb_dir,
        collection_name = 'parent_child_docs'
        ) -> None:
        if chromdb_dir:
            client = chromadb.PersistentClient(path=chromdb_dir, settings = Settings(anonymized_telemetry=False))
        else:
            client = chromadb.Client(settings = Settings(anonymized_telemetry=False))
        self.chroma_collection = client.get_or_create_collection(name = collection_name)
        self.parent_docs_store = {}
        self.text_splitter = RecursiveSplitter()
        
    def _ingest_parent_docs(self, state):
        parent_chunks = self.text_splitter.split_text(state.documents_text, state.chunk_size, state.chunk_overlap)
        total_chunks_size = 0
        for chunk in tqdm(parent_chunks):
            # Create a consistent ID using a hash of the chunk content
            parent_id = hashlib.sha256(chunk.encode()).hexdigest()
            self.parent_docs_store[parent_id] = chunk
            total_chunks_size += len(chunk)
        print(f'INFO: Created and stored {len(self.parent_docs_store)} parent chunks.')
        return total_chunks_size // (len(parent_chunks))
         
    def _ingest_child_docs(self, state):
        """
        Splits parent chunks into child chunks, embeds them, and stores them in ChromaDB.
        """
        if not self.parent_docs_store:
            print('ERROR: No parent chunks found. Ingestion cannot proceed.')
            raise ValueError('\nParent chunks were not created. Please ingest a document first.')

        total_chunk_size = 0
        for parent_id, parent_chunk in tqdm(self.parent_docs_store.items()):
            child_chunks = self.text_splitter.split_text(parent_chunk, state.chunk_size // 2,
                                            state.chunk_overlap // 2)
            
            for i, child_chunk in enumerate(child_chunks):
                child_id = hashlib.sha256(child_chunk.encode()).hexdigest()
                
                # generate embeddings
                embeddings = ollama.embed(state.embed_model, child_chunk)['embeddings']
                self.chroma_collection.add(
                    ids = child_id,
                    embeddings = embeddings,
                    metadatas={"parent_id": parent_id, "chunk_index": i},
                    documents=child_chunk
                )
                
                total_chunk_size += len(child_chunk)
        print(f'INFO: Added {self.chroma_collection.count()} child chunks to ChromaDB.')
        return total_chunk_size / self.chroma_collection.count()
                
    def _ingest_documents(self, state) :
        try: 
            # process parent chunks
            print('INFO: Ingesting parent chunks...')
            start = time.time()
            avg_parent_size = self._ingest_parent_docs(state)
            parent_pro_time = time.time() - start
            
            # process child chunks
            print('INFO: Ingesting child chunks...')
            start = time.time()
            avg_child_size = self._ingest_child_docs(state)
            print('INFO: Document successfully ingested into parent-child structure.')
            child_pro_time = time.time() - start
            
            # get counts and sizes for metadata
            num_parent_chunks = len(self.parent_docs_store)
            num_child_chunks = self.chroma_collection.count()
            total_pro_time = parent_pro_time + child_pro_time
            
            metadata = {
            'Number of Parent Chunks': num_parent_chunks,
            'Number of Child Chunks': num_child_chunks,
            'Parent Ingestion Time': round(parent_pro_time, 2),
            'Child Ingestion Time': round(child_pro_time, 2),
            'Total Ingestion Time': round(total_pro_time, 2),
            'Average Parent Chunk Size': round(avg_parent_size, 2),
            'Average Child Chunk Size': round(avg_child_size, 2),
            'Child to Parents Chunk Ratio': round(num_child_chunks / num_parent_chunks, 2)
            }
            
            max_key_length = max(len(k) for k in metadata.keys())
            metadata_text = ''
            for k, v in metadata.items():
                metadata_text += f'{k:<{max_key_length+10}} {v}\n'
            return '**Summary:**\n' +f'```{metadata_text}```'
            
        except Exception as e:
            print(f'ERROR: A critical error occurred during document ingestion: {e}')
            return 'An error occured during document ingestion.'
        
    def _retrieve_docs(self, query, state):
        if not self.parent_docs_store:
            print("ERROR: No parent documents are stored in memory. Retrieval will fail.")
            return ''
        
        try:
            query_embeddings = ollama.embed(state.embed_model, query)['embeddings']
            query_results = self.chroma_collection.query(
                query_embeddings=query_embeddings,
                n_results=state.n_results,
                include=['metadatas']
            )

            retrieved_parent_ids = set()
            if query_results['metadatas']:
                for metadata_lists in query_results['metadatas']:
                    for metadata in metadata_lists:
                        if 'parent_id' in metadata:
                            retrieved_parent_ids.add(metadata['parent_id'])
            
            retrieved_parent_docs = []
            for parent_id in retrieved_parent_ids:
                if parent_id in self.parent_docs_store:
                    retrieved_parent_docs.append(self.parent_docs_store[parent_id])
            
            context = ''
            for doc in retrieved_parent_docs:
                context += f'- {doc}\n'
            return context
        
        except Exception as e:
            print(f'ERROR: An error occured during document retrieval: {e}')
            return ''

class MapReduceSummarizer:
    def __init__(self) -> None:
        self.text_splitter = RecursiveSplitter()
        self.docs_reader = PdfReader()
        
    def _map_function(self, chunk, state):
        map_prompt = (
            'You are a helpful assistant. Please summarize the following text. '
            'Keep the summary concise and retain the most important details.\n\n'
            f'TEXT:\n---\n{chunk}\n---\nSUMMARY: '
        )
        response = ollama.generate(model=state.chat_model, prompt=map_prompt)['response']
        return response
    
    def _reduce_function(self, summaries, state):
        reduce_prompt = (
            'You are a helpful summarization assistant. You are given several summaries of a long document. '
            'Please combine these summaries into one final, coherent summary. '
            'Ensure all key points from the indivual summaries are included. '
            'Output the summary in proper Markdown format with bulltet key points.\n\n'
            f'SUMMARIES:\n---\n{'\n- '.join(summaries)}\n---\nFINAL SUMMARY: '
        )
        response = ollama.generate(model=state.chat_model, prompt=reduce_prompt)['response']
        return response

    
    def _pre_summarize(self, state, chunk_size, chunk_overlap):
        if not state.documents_text:
            print('ERROR: Read document first.')
            raise ValueError('No Text was read.')
        
        chunks = self.text_splitter.split_text(state.documents_text, chunk_size, chunk_overlap)
        avg_chunk_len = sum([len(c) for c in chunks]) // len(chunks)
        
        metadata = {
            'num_chunks': len(chunks),
            'avg_chunk_len': avg_chunk_len
        }
        
        max_key_length = max(len(k) for k in metadata.keys())
        metadata_text = ''
        for k, v in metadata.items():
            metadata_text += f'{k:<{max_key_length+10}} {v}\n'
        return f'```\n{metadata_text}\n```'
        
    def summarize(self, state, chunk_size, chunk_overlap):
        if not state.documents_text:
            print('ERROR: Read document first.')
            raise ValueError('No document was read.')
        print('INFO: Starting summarization...')
        chunks = self.text_splitter.split_text(state.documents_text[:2000], chunk_size, chunk_overlap)
        
        chunk_summaries = []
        total_chunk_len = 0
        print(f'INFO: Summarizing Chunks...')
        for chunk in tqdm(chunks):
            summary = self._map_function(chunk, state)
            if summary:
                chunk_summaries.append(summary)
            
            chunk_len = len(chunk)
            total_chunk_len += chunk_len
        if not chunk_summaries:
            print('WARNING: No summaries were generated in Mapping step.')
            return '```\nFailed to summarize document.\n```'
        
        print(f'INFO: Consolidating Chunk Summaries.')
        final_summary = self._reduce_function(chunk_summaries, state)
        state.sum_text = final_summary
        print('INFO: Finished Summarization.')
        return final_summary

    def save_summary_markdown(self, filename, state):

        if not state.sum_text:
            print('ERROR: Generate summary first', 'e')
            raise ValueError('Text was not summarized.')

        # Remove trailing whitespaces
        basename = filename.strip()
        
        # Check if the stripped filename is empty
        if not basename:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            basename = f'Summary_{timestamp}'
        
        # Ensure the extension is .md
        if not basename.endswith('.md'):
            basename += '.md'

        # Ensure chats folder exists
        os.makedirs(state.sum_dir, exist_ok=True)
        filepath = os.path.join(state.sum_dir, basename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(state.sum_text)
            print(f'INFO: Summary successfully saved to: {filepath}')
            return f'Saved summary to: {filepath}'
        except Exception as e:
            print(f'ERROR: Failed to save markdown file: {e}')
            return 'Failed to save summary file.'
            
class WebAgent:
    def _rephrase_query(self, query, chat_history, state):
        now = datetime.now().strftime('%Y-%m-%d')
        
        rephrase_system_prompt = (
            "You are a helpful assistant. Your role is to rewrite a given query "
            "in three different ways to fetch the most **updated** results from a search engine.\n"
            f"**Current Date**: {now}. You can use the date when forming queries.\n"
            "Provide the output in JSON format."
            '```json\n{"queries": ["query1", "query2", "query3"]}\n```'
        )
        ollama_messages = (
            [{'role': 'system', 'content': rephrase_system_prompt}] +
            chat_history +
            [{'role': 'user', 'content': query}]
        )
        
        response = ollama.chat(
            state.chat_model, 
            ollama_messages, 
            options={'temperature': 1.0},
            )['message']['content']
    
        try:
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            data = json.loads(json_str)
            queries = data.get("queries", [])
            return queries
            
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON from LLM response. Error: {e}. Raw response: {response}")
            return [query]
        except (KeyError, IndexError) as e:
            print(f"ERROR: Unexpected JSON structure or key not found. Error: {e}")
            return [query]

    def _get_domains_suggestions(self, query, shared_state):
        sites_suggester_system_prompt = (
        'Your role is to provide a list of three trustworthy domains which may have relevant data to user query.'
        'Provide output in JSON format.'
        '```json\n{"suggestions": ["wwww.domain1.com", "www.domain2.com", "www.domain3.com"]}\n```'
        )
        ollama_messages = [
            {'role': 'system', 'content': sites_suggester_system_prompt},
            {'role': 'user', 'content': query}
        ]

        response = ollama.chat(
            model=shared_state.chat_model,
            messages=ollama_messages
        )['message']['content']

        try:
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON code block found in the LLM response.")

            json_str = json_match.group(1)
            data = json.loads(json_str)
            suggestions = data.get("suggestions", [])
            if not all(isinstance(s, str) for s in suggestions):
                raise TypeError("Suggestions must be a list of strings.")
            return suggestions
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"ERROR: Failed to process LLM response for domain suggestions. Error: {e}. Raw response: {response}")
            return []
    
    def _search_web_chain(self, user_msg, history, state):
        metadata = {'context': '', 'sources': set(), 'rephrased_queries': []}
        
        print(f'INFO: Performing web searches...')
        # 1. Generate the intial rephrased query
        rephrased_queries = self._rephrase_query(user_msg, history, state)
        
        #2. Perform a general search first
        for query in rephrased_queries:
            try:
                web_search_data = DDGS().text(query=query, max_results = 3)
                for search in web_search_data:
                    context_text = f'\n- ({search.get("href", "N/A")})\t{search.get("body", "No body text available.")}'
                    metadata['context'] += context_text
                    metadata['sources'].add(search.get('href', 'N/A'))
            except Exception as e:
                print(f"ERROR: Failed to perform search for query '{query}': {e}")
                continue
            
        # 4. Perform targeted searches for each suggested domain
        #sites_suggestion = self._get_domains_suggestions(rephrased_queries[0], shared_state)
        # for site in sites_suggestion:
        #     if site:
        #         targeted_query = f'{rephrased_queries} {site}'
        #         metadata['rephrased_queries'].append(targeted_query)
        #         web_searches_targeted = DDGS().text(query=targeted_query, max_results = 3)
        #         for search in web_searches_targeted:
        #             metadata['context'] += '\n- ' + f'({search['href']})\t' + search['body']
        #             metadata['sources'].add(search['href'])
        print(f'INFO: Compeleted web searches.')
        metadata['rephrased_queries'] = rephrased_queries
        return metadata  

class GenPipe:
    def __init__(self):
        self.text_ingester = ParentChildsplitter()
        self.web_agent = WebAgent()
        
    def _ingest_docs(self, state):
        # want to ensure that parent doc store (and vectorstore if in memory) is available for retrieval
        summary = self.text_ingester._ingest_documents(state)
        return summary
    
    def _filter_metadata(self, history):
        ollama_history = []
        for msg in history:
            if not msg.get('metadata'):
                ollama_history.append(msg)
        return ollama_history
    
    def _prepate_msg(self, msg, history, context, state):
        history = self._filter_metadata(history)
        
        if state.rag_flag is True:
            system_prompt = 'You are a helpful assistant who answers questions based **only** on the provided context.'
        elif state.ddgs_flag is True:
            system_prompt = (
                'You are provided with a main query, sub-queries, and answers to sub-queries from a web search. '
                'Your role is to retrieve the answer from the complete text. Be extra careful as the search '
                'results might have conflicting new and old data. Synthesize a coherent answer.'
            )
        else:
            system_prompt = state.system_prompt            
        
        if state.user_name:
            msg = f'{state.user_name}: ' + msg
        
        if context:
            msg += '\nContext: ' + context
        
        if state.bot_name: 
            msg = msg + f'\n{state.bot_name}: '
        
        ollama_messages = (
            [{'role': 'system', 'content': system_prompt}] + 
            history + 
            [{'role': 'user', 'content': msg}]
        )
        return ollama_messages
         
    def _ollama_response(self, msg, history, state):
        context = ''
        metadata = {}
        if state.rag_flag is True:
            context = self.text_ingester._retrieve_docs(msg, state)
        elif state.ddgs_flag is True:
            metadata = self.web_agent._search_web_chain(msg, history, state)
            context = metadata['context']
        
        # get formatted ollama message fed to llm
        ollama_messages = self._prepate_msg(msg, history, context, state)
        
        # update history with user message
        history += [{'role': 'user', 'content': msg}]
        # create a bot message placeholder for streaming
        history += [{'role': 'assistant', 'content': ''}]
        
        # stream llm message
        full_res = ''
        res_stream = ollama.chat(
            model=state.chat_model, 
            messages=ollama_messages,
            stream=True,
            options= {
                'temperature': float(state.temperature),
                'top_k': state.top_k,
                'top_p': state.top_p,
                'num_ctx': state.num_ctx
            }
            )
        
        for i in tqdm(res_stream, 'LLM Generation...'):
            full_res += i['message']['content']
            history[-1]['content'] = full_res
            yield '', history, history
            
        
        if context:
            if state.ddgs_flag is True:
                sources_str = ', '.join(metadata.get('sources', []))
                context_with_sources = (f'**Web Search Results**:\n\n{context}\n\n**Sources**:\n\n{sources_str}')
                history.append({'role': 'assistant', 'content': context_with_sources, 'metadata': {'title': 'Web Search'}})
            elif state.rag_flag is True:
                history.append({'role': 'assistant', 'content': context, 'metadata': {'title': 'Web Search'}})
            else:
                pass
        yield '', history, history
    
    def _regen_msg(self, history, state):
        msg = ''
        while(len(history)) > 0:
            last_msg = history.pop()
            if last_msg.get('role', '') == 'user':
                msg = last_msg['content']
                break
        if msg:
            yield from self._ollama_response(msg, history, state)

class ChatUI:
    def __init__(self, state: gr.State) -> None:
        self.state = state
        self.history = gr.State([])
        
        self.file_man = FileManager()
        self.pdf_reader = PdfReader()
        self.gen_pipe = GenPipe()
        self.summarizer = MapReduceSummarizer()
        self._interface()
        self._events()
    
    def _interface(self):
        # sidebar defintion
        with gr.Sidebar():
            gr.Markdown('Navigation')
            self.chat_tab_btn = gr.Button('Chat')
            self.docs_tab_btn = gr.Button('Documents')
            self.settings_tab_btn = gr.Button('Settings')
        
        with  gr.Column(visible=True) as self.chat_page:
            gr.Markdown('## Chat')        
            with gr.Row():
                self.chat_drop = gr.Dropdown(
                    [f for f in os.listdir(self.state.value.chats_dir) if f.endswith('json')],
                    label='File to load',
                    interactive=True
                    )
                self.chat_name_in = gr.Textbox(label='New Chat', interactive=True)

            with gr.Row():
                self.save_chat_btn = gr.Button('Save Chat File', size='md')
                self.load_chat_btn = gr.Button('Load Chat File', size='md')
            
            self.chatbot = gr.Chatbot(
            show_label=False,
            type='messages',
            height=450
            )

            with gr.Row():
                self.clear_msg_btn = gr.Button(value='Clear Complete Chat', interactive=True)
                self.ddgs_flag_in = gr.Checkbox(value=False, label='Duckduckgo Search')
                self.rag_flag_in = gr.Checkbox(value=False, label='RAG')
                self.regen_msg_btn = gr.Button(value='Regenerate last message', interactive=True)

            self.user_msg_in = gr.Textbox(
            lines=1, 
            scale=3,
            interactive=True,
            submit_btn=True,
            stop_btn=True
            )
            
        with  gr.Column(visible=False) as self.docs_page:
            gr.Markdown('## Documents')   
            with gr.Tab('Reading'):
                gr.Markdown('### Reading PDF files.')
                self.pdf_files_in = gr.File(
                    file_count='multiple',
                    file_types=['.pdf'],
                    label='PDF File',
                    height=150
                )
                self.read_pdfs_btn = gr.Button('Read PDFs', interactive=True)
                self.pdfs_status = gr.Markdown('``````')
            
            with gr.Tab('RAG'):
                gr.Markdown('### Preprocessing text for RAG.')
                with gr.Row():
                    self.chunk_size_in = gr.Slider(0, 512, 450, step=1, label='Chunk Size', interactive=True)
                    self.chunk_overlap_in = gr.Slider(0, 64, 45, step=1, label='Chunk Overlap', interactive=True)
                    
                self.rag_btn = gr.Button('Preprocess', interactive=True)
                
                with gr.Row():
                    self.rag_status = gr.Markdown('``````')
                
            with gr.Tab('Summarization'):
                gr.Markdown('### Summarizing documents.')
                with gr.Row():
                    self.chunk_size_s_in = gr.Slider(0, 8192, 1800, step=1, label='Chunk Size', interactive=True)
                    self.chunk_overlap_s_in = gr.Slider(0, 1024, 180, step=1, label='Chunk Overlap', interactive=True)
                    
                with gr.Row():
                    self.pre_sum_btn = gr.Button('Pre-Summarization info', interactive=True)
                    self.sum_btn = gr.Button('Summarize', interactive=True)
                    self.save_sum_btn = gr.Button('Save summary', interactive=True)
                
                with gr.Row():
                    self.sum_save_file_in = gr.Textbox(label='Summary File Name', interactive=True)
                    self.sum_save_status = gr.Textbox(label='Save status', interactive=False)
                    
                gr.Markdown('### Pre Summarization Status')
                self.pre_sum_status = gr.Markdown('``````')
                gr.Markdown('### Summarized Documents')
                with gr.Group():
                    self.sum_status = gr.Markdown('``````')
            
        with gr.Column(visible=False) as self.settings_page:
            gr.Markdown('## Settings')
            with gr.Tab(label='Model Settings'):
                available_models = _get_ollama_models()
                gr.Markdown('**Model Options**')
                with gr.Row():
                    self.chat_model_in = gr.Dropdown(
                        choices=available_models,
                        label='Carefully choose Chat model.',
                        interactive=True,
                        value=available_models[0]
                    )
                    self.emb_model_in = gr.Dropdown(
                        choices=available_models,
                        label='Carefully choose Embedding model.',
                        interactive=True,
                        value=available_models[0]
                    )
                gr.Markdown('**System Prompt**')
                self.system_prompt_in = gr.Textbox(
                    value=self.state.value.system_prompt,
                    label='System Prompt',
                    lines=5,
                    interactive=True
                )
                
                gr.Markdown('**Chat Character Names**')
                with gr.Row():
                    self.user_name_in = gr.Textbox(label='User Name', interactive=True)
                    self.bot_name_in = gr.Textbox(label='Bot Name', interactive=True)

                gr.Markdown('**Sampling Parameters**')
                with gr.Row():
                    self.temeprature_in = gr.Slider(minimum=0, maximum=2, step=0.05, value=1.0,label='Temperature', interactive=True)
                    self.top_k_in = gr.Slider(minimum=1, maximum=100, step=1, value=64, label='Top-K', interactive=True)
                    self.top_p_in = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.95, label='Top-P', interactive=True)
                    self.n_ctx_in = gr.Slider(minimum=10, maximum=18, step=1, value=11, label='2^(n) Context Length', interactive=True)
                    
                self.update_model_settings_btn = gr.Button(value='Update Model Settings', size='md', interactive=True)
            with gr.Tab(label='RAG Settings'):
                self.n_results_in = gr.Slider(0, 10, step=1, value=5, label='Number of Retrieved Results', interactive=True)
                self.chromadb_dir_in = gr.Textbox(label='Chromadb store directory', interactive=True)

                self.update_rag_settings_btn = gr.Button(value='Update Rag Settings', interactive=True)
            self.settings_page_status = gr.Markdown(value='``````', label='Settings Status', visible=True)
    
    def _events(self):
        # sidebar events
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
        
        self.user_msg_in.submit(
            self.gen_pipe._ollama_response,
            [self.user_msg_in, self.history, self.state],
            [self.user_msg_in, self.history, self.chatbot]
        )
        self.clear_msg_btn.click(
            lambda: ([], []),
            outputs=[self.chatbot, self.history],
        )
        self.regen_msg_btn.click(
            self.gen_pipe._regen_msg,
            [self.history, self.state],
            [self.user_msg_in, self.history, self.chatbot]
        )
        self.ddgs_flag_in.change(
            self.state.value._toggle_ddgs_flag,
            [self.ddgs_flag_in],
            [self.state]
        )
        self.rag_flag_in.change(
            self.state.value._toggle_rag_flag,
            [self.rag_flag_in],
            [self.state]
        )
        self.load_chat_btn.click(
            self.file_man._load_chat,
            [self.chat_drop, self.state],
            [self.history, self.chatbot, self.state, self.chat_name_in]
        )
        self.save_chat_btn.click(
            self.file_man._save_chat,
            [self.chat_name_in, self.history, self.state],
            [self.chat_drop, self.chat_name_in]
        )
        
        self.read_pdfs_btn.click(
            self.pdf_reader._read_pdfs,
            [self.pdf_files_in, self.state],
            [self.state, self.pdfs_status]
        )
        self.rag_btn.click(
            self.gen_pipe._ingest_docs,
            [self.state],
            [self.rag_status]
        )
        self.pre_sum_btn.click(
            self.summarizer._pre_summarize,
            [self.state, self.chunk_size_s_in, self.chunk_overlap_s_in],
            [self.pre_sum_status]
        )
        self.sum_btn.click(
            self.summarizer.summarize,
            [self.state, self.chunk_size_s_in, self.chunk_overlap_s_in],
            [self.sum_status]
        )
        self.save_sum_btn.click(
            self.summarizer.save_summary_markdown,
            [self.sum_save_file_in, self.state],
            [self.sum_save_status]
        )
        self.update_model_settings_btn.click(
            self.state.value._update_model_settings,
            [
                self.chat_model_in, self.emb_model_in, 
                self.system_prompt_in, self.temeprature_in,
                self.top_k_in, self.top_p_in,
                self.n_ctx_in, self.user_name_in, self.bot_name_in
             ],
            [self.state, self.settings_page_status]
        )
        self.update_rag_settings_btn.click(
            self.state.value._update_rag_settings,
            [self.n_results_in, self.chromadb_dir_in],
            [self.state, self.settings_page_status ]
        )
    
with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    ChatUI(app_shared_state)

demo.launch() 