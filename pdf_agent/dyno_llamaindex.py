from typing import List, Dict, Any, Optional, Union, Callable
import os
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex
)
from llama_index.readers.file import PDFReader, DocxReader, CSVReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SimpleNodeParser

# Import ChromaDB only when it's used
def import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        return None

class DynoDataLoader:
    """DynoFrame specialized data loader for various file types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chromadb = import_chromadb()
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from various file types."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            reader = PDFReader()
        elif file_ext == '.docx':
            reader = DocxReader()
        elif file_ext == '.csv':
            reader = CSVReader()
        elif file_ext in ['.html', '.htm']:
            reader = SimpleWebPageReader()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return reader.load_data(file_path)

    def register_custom_loader(self, source_type: str, loader_func: Callable) -> None:
        """Register a custom loader function for a specific source type."""
        self.custom_loaders[source_type] = loader_func
    
    def load_data(self, source_type: str, source_data: Any) -> List[Document]:
        """Load data from different source types."""
        if source_type in self.custom_loaders:
            # Use custom loader if registered
            documents = self.custom_loaders[source_type](source_data)
            self.loaded_data.extend(documents)
            self.data_sources[source_type] = source_data
            return documents
        
        if source_type == "directory":
            documents = self._load_from_directory(source_data)
        elif source_type == "pdf":
            documents = self._load_from_pdf(source_data)
        elif source_type == "docx":
            documents = self._load_from_docx(source_data)
        elif source_type == "csv":
            documents = self._load_from_csv(source_data)
        elif source_type == "webpage":
            documents = self._load_from_webpage(source_data)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        self.loaded_data.extend(documents)
        self.data_sources[source_type] = source_data
        return documents
    
    def _load_from_directory(self, directory_paths: List[str]) -> List[Document]:
        """Load documents from directories."""
        documents = []
        for directory in directory_paths:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist.")
                continue
            
            reader = SimpleDirectoryReader(directory)
            docs = reader.load_data()
            documents.extend(docs)
        
        return documents
    
    def _load_from_pdf(self, pdf_paths: List[str]) -> List[Document]:
        """Load documents from PDF files."""
        reader = PDFReader()
        documents = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file {pdf_path} does not exist.")
                continue
            
            docs = reader.load_data(pdf_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_docx(self, docx_paths: List[str]) -> List[Document]:
        """Load documents from DOCX files."""
        reader = DocxReader()
        documents = []
        
        for docx_path in docx_paths:
            if not os.path.exists(docx_path):
                print(f"Warning: DOCX file {docx_path} does not exist.")
                continue
            
            docs = reader.load_data(docx_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_csv(self, csv_paths: List[str]) -> List[Document]:
        """Load documents from CSV files."""
        reader = CSVReader()
        documents = []
        
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file {csv_path} does not exist.")
                continue
            
            docs = reader.load_data(csv_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_webpage(self, urls: List[str]) -> List[Document]:
        """Load documents from web pages."""
        reader = SimpleWebPageReader()
        documents = reader.load_data(urls)
        return documents
    
    def create_index(self) -> VectorStoreIndex:
        """Create a vector store index from loaded documents."""
        if not self.loaded_data:
            raise ValueError("No documents loaded. Please load documents first.")
        
        # If ChromaDB is enabled and available, use it
        if self.use_chromadb and hasattr(self, 'chroma_client') and self.chroma_client is not None:
            try:
                # Ensure we have a valid ChromaDB collection
                if self.chroma_collection is None:
                    try:
                        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
                    except Exception as e:
                        print(f"Warning: Could not create ChromaDB collection: {str(e)}")
                        self.use_chromadb = False  # Fall back to default
                
                if self.chroma_collection is not None:
                    # Create the vector store and storage context with ChromaDB
                    vector_store = self.ChromaVectorStore(chroma_collection=self.chroma_collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    # Create the index with ChromaDB vector store
                    self.index = VectorStoreIndex.from_documents(
                        documents=self.loaded_data,
                        storage_context=storage_context
                    )
                    print(f"Created index using ChromaDB collection: {self.collection_name}")
                    return self.index
            except Exception as e:
                print(f"Error creating index with ChromaDB: {str(e)}")
                print("Falling back to default in-memory vector store...")
                self.use_chromadb = False  # Fall back to default
        
        # Default in-memory vector store
        try:
            # Try the new way (with settings)
            self.index = VectorStoreIndex.from_documents(
                documents=self.loaded_data
            )
        except TypeError:
            # Fall back to the old way (with service_context)
            self.index = VectorStoreIndex.from_documents(
                documents=self.loaded_data,
                service_context=self.service_context
            )
        
        print("Created index using default in-memory vector store")
        return self.index
    
    def persist_index(self) -> None:
        """Persist the index to disk."""
        if self.index is None:
            raise ValueError("No index created. Please create an index first.")
        
        # If using ChromaDB, note that it's already being persisted automatically
        if self.use_chromadb and hasattr(self, 'chroma_collection') and self.chroma_collection is not None:
            print(f"ChromaDB collection '{self.collection_name}' is being used and automatically persisted to {self.db_path}")
        
        # Create the directory if it doesn't exist
        persist_path = os.path.join(self.persist_dir, "index")
        os.makedirs(persist_path, exist_ok=True)
        
        # Persist the index using storage context
        storage_context = self.index.storage_context
        storage_context.persist(persist_dir=persist_path)
        print(f"Index persisted to {persist_path}")
    
    def load_index(self, index_path: Optional[str] = None) -> VectorStoreIndex:
        """Load an index from disk."""
        index_path = index_path or os.path.join(self.persist_dir, "index")
        
        # First try loading from ChromaDB if enabled and available
        if self.use_chromadb and hasattr(self, 'chroma_client') and self.chroma_client is not None:
            try:
                # Check if ChromaDB collection exists
                chroma_collection = self.chroma_client.get_collection(self.collection_name)
                self.chroma_collection = chroma_collection
                
                # Create vector store and load index
                vector_store = self.ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Load index from vector store
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                )
                print(f"Successfully loaded index from ChromaDB collection: {self.collection_name}")
                return self.index
            except Exception as e:
                print(f"Could not load index from ChromaDB: {str(e)}")
                print("Trying to load from disk storage...")
        
        # Fall back to loading from disk storage
        if not os.path.exists(index_path):
            raise ValueError(f"Index path '{index_path}' does not exist.")
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            self.index = load_index_from_storage(storage_context)
            return self.index
        except Exception as e:
            raise ValueError(f"Failed to load index from {index_path}: {str(e)}")
    
    def query_index(self, query_text: str) -> Dict[str, Any]:
        """Query the index with a text query."""
        if self.index is None:
            raise ValueError("No index available. Please create or load an index first.")
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        
        return {
            "answer": str(response),
            "source_nodes": [n.node.get_text() for n in response.source_nodes]
        } 