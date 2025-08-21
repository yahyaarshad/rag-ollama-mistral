from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss


class DocumentIndexer:
    _instance = None  # Singleton instance holder

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # To avoid reinitialization on multiple calls
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.documents = self.splitter.split_text(
            """
            This Retrieval-augmented generation (RAG) based Chat bot is developed by Yahya Arshad.
            This Software uses FAISS, and Ollama Mistral.
            """
        )
        self.embeddings = self.embedder.encode(self.documents, convert_to_numpy=True).astype('float32')
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)  # Add initial embeddings to FAISS index

        self._initialized = True  # Flag to prevent reinitialization

    def index_new_document(self, document_str):
        new_doc = self.splitter.split_text(document_str)
        new_embeddings = self.embedder.encode(new_doc, convert_to_numpy=True).astype('float32')
        self.index.add(new_embeddings)  # Add to FAISS index
        self.documents.extend(new_doc)

    def semantic_search(self, query, top_k=3):
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        return "\n".join([self.documents[i] for i in indices[0]])
