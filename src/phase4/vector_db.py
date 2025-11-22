"""Vector Database Manager using ChromaDB."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from config.settings import DATA_DIR, PROCESSED_DATA_DIR, GUIDES_DATA_DIR, EMBEDDING_MODEL, CHROMA_DATA_DIR

logger = setup_logger(__name__)


class VectorDBManager:
    """Manages vector database operations for RAG system."""
    
    def __init__(self, db_path: Optional[Path] = None, embedding_model: Optional[str] = None):
        """Initialize Vector Database Manager.
        
        Args:
            db_path: Path to ChromaDB storage (default: data/chroma)
            embedding_model: Sentence transformer model name (default: from settings)
        """
        # Use embedding model from settings if not provided
        if embedding_model is None:
            embedding_model = EMBEDDING_MODEL
        self.db_path = db_path or CHROMA_DATA_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self.db_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize collections
        self.tickets_collection = None
        self.guides_collection = None
        
    def create_collections(self, reset: bool = False):
        """Create or get collections for tickets and guides.
        
        Args:
            reset: If True, delete existing collections and create new ones
        """
        if reset:
            logger.warning("Resetting collections - all existing data will be deleted")
            try:
                self.client.delete_collection("tickets")
                self.client.delete_collection("guides")
                logger.info("Deleted existing collections")
            except Exception as e:
                logger.debug(f"No existing collections to delete: {e}")
        
        # Create tickets collection
        self.tickets_collection = self.client.get_or_create_collection(
            name="tickets",
            metadata={
                "description": "Historical Zendesk support tickets",
                "hnsw:space": "cosine"
            }
        )
        logger.info(f"Tickets collection ready: {self.tickets_collection.count()} documents")
        
        # Create guides collection
        self.guides_collection = self.client.get_or_create_collection(
            name="guides",
            metadata={
                "description": "LaCuraDellAuto technical guides",
                "hnsw:space": "cosine"
            }
        )
        logger.info(f"Guides collection ready: {self.guides_collection.count()} documents")
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        # Handle both numpy arrays and lists
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings
    
    def add_tickets(self, tickets_file: Optional[Path] = None, batch_size: int = 50):
        """Load and add tickets to vector database.
        
        Args:
            tickets_file: Path to processed tickets JSON
            batch_size: Number of tickets to process at once
        """
        tickets_file = tickets_file or PROCESSED_DATA_DIR / "processed_tickets.json"
        
        if not tickets_file.exists():
            logger.error(f"Tickets file not found: {tickets_file}")
            raise FileNotFoundError(f"Tickets file not found: {tickets_file}")
        
        logger.info(f"Loading tickets from: {tickets_file}")
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        logger.info(f"Processing {len(tickets)} tickets")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for ticket in tickets:
            ticket_id = f"ticket_{ticket['ticket_id']}"
            
            # Use searchable_text as the document
            document = ticket.get('searchable_text', '')
            if not document:
                logger.warning(f"Ticket {ticket['ticket_id']} has no searchable text, skipping")
                continue
            
            # Metadata (ChromaDB doesn't accept None values)
            metadata = {
                'ticket_id': str(ticket['ticket_id']),
                'subject': str(ticket.get('subject', ''))[:500],  # Limit length
                'status': str(ticket.get('status', '')),
                'priority': str(ticket.get('priority', '')),
                'created_at': str(ticket.get('created_at', '')),
                'updated_at': str(ticket.get('updated_at', '')),
                'comment_count': int(ticket.get('comment_count', 0)),
                'type': 'ticket'
            }
            
            # Remove empty string values to save space
            metadata = {k: v for k, v in metadata.items() if v not in ['', 'None', None]}
            
            ids.append(ticket_id)
            documents.append(document)
            metadatas.append(metadata)
        
        # Generate embeddings and add to collection
        logger.info(f"Generating embeddings for {len(documents)} tickets")
        embeddings = self.generate_embeddings(documents)
        
        logger.info(f"Adding {len(documents)} tickets to vector database")
        self.tickets_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(documents)} tickets to vector database")
        
    def add_guides(self, guides_file: Optional[Path] = None, batch_size: int = 50):
        """Load and add guide sections to vector database.
        
        Args:
            guides_file: Path to guides JSON
            batch_size: Number of sections to process at once
        """
        guides_file = guides_file or GUIDES_DATA_DIR / "guides.json"
        
        if not guides_file.exists():
            logger.error(f"Guides file not found: {guides_file}")
            raise FileNotFoundError(f"Guides file not found: {guides_file}")
        
        logger.info(f"Loading guides from: {guides_file}")
        with open(guides_file, 'r', encoding='utf-8') as f:
            guides = json.load(f)
        
        logger.info(f"Processing {len(guides)} guides")
        
        # Prepare data for ChromaDB (each section is a separate document)
        ids = []
        documents = []
        metadatas = []
        
        for guide in guides:
            guide_number = guide.get('guide_number', 'UNKNOWN')
            guide_title = guide.get('title', '').strip()
            guide_url = guide.get('url', '')
            
            for idx, section in enumerate(guide.get('sections', [])):
                section_title = section.get('title', '')
                section_content = section.get('content', '')
                
                if not section_content or len(section_content) < 50:
                    logger.debug(f"Skipping section with insufficient content: {section_title}")
                    continue
                
                # Create document combining section title and content
                document = f"{section_title}\n\n{section_content}"
                
                # Unique ID for this section
                section_id = f"guide_{guide_number}_{idx}"
                
                # Metadata (ChromaDB doesn't accept None values)
                metadata = {
                    'guide_number': str(guide_number),
                    'guide_title': str(guide_title)[:500],
                    'section_title': str(section_title)[:500],
                    'section_index': int(idx),
                    'url': str(guide_url),
                    'anchor_id': str(section.get('anchor_id', '')),
                    'content_length': int(section.get('content_length', 0)),
                    'type': 'guide'
                }
                
                # Remove empty string values to save space
                metadata = {k: v for k, v in metadata.items() if v not in ['', 'None', None]}
                
                ids.append(section_id)
                documents.append(document)
                metadatas.append(metadata)
        
        # Generate embeddings and add to collection
        logger.info(f"Generating embeddings for {len(documents)} guide sections")
        embeddings = self.generate_embeddings(documents)
        
        logger.info(f"Adding {len(documents)} guide sections to vector database")
        self.guides_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(documents)} guide sections to vector database")
    
    def search_tickets(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant tickets.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with documents, metadata, and distances
        """
        logger.debug(f"Searching tickets for: {query[:100]}...")
        
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.tickets_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        logger.debug(f"Found {len(results['ids'][0]) if results['ids'] else 0} ticket results")
        return results
    
    def search_guides(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant guide sections.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with documents, metadata, and distances
        """
        logger.debug(f"Searching guides for: {query[:100]}...")
        
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.guides_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        logger.debug(f"Found {len(results['ids'][0]) if results['ids'] else 0} guide results")
        return results
    
    def search_all(self, query: str, n_tickets: int = 3, n_guides: int = 3) -> Dict[str, Any]:
        """Search both tickets and guides.
        
        Args:
            query: Search query
            n_tickets: Number of ticket results
            n_guides: Number of guide results
            
        Returns:
            Combined search results
        """
        logger.info(f"Searching all sources for: {query[:100]}...")
        
        ticket_results = self.search_tickets(query, n_tickets)
        guide_results = self.search_guides(query, n_guides)
        
        return {
            'query': query,
            'tickets': ticket_results,
            'guides': guide_results
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics.
        
        Returns:
            Dictionary with collection counts
        """
        return {
            'tickets': self.tickets_collection.count() if self.tickets_collection else 0,
            'guides': self.guides_collection.count() if self.guides_collection else 0
        }

