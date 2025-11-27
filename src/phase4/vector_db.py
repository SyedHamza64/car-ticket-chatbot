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
        
        # Create/get tickets collection
        self.tickets_collection = self.client.get_or_create_collection(
            name="tickets",
            metadata={
                "description": "Historical Zendesk support tickets",
                "hnsw:space": "cosine"
            }
        )
        
        # Create/get guides collection
        self.guides_collection = self.client.get_or_create_collection(
            name="guides",
            metadata={
                "description": "Product guides and technical documentation",
                "hnsw:space": "cosine"
            }
        )
        
        logger.info(f"Collections ready - tickets: {self.tickets_collection.count()}, guides: {self.guides_collection.count()}")
        
    def reset_tickets_collection(self):
        """Reset only the tickets collection."""
        logger.info("Resetting tickets collection...")
        try:
            self.client.delete_collection("tickets")
            logger.info("Deleted tickets collection")
        except Exception as e:
            logger.debug(f"No tickets collection to delete: {e}")
        
        self.tickets_collection = self.client.get_or_create_collection(
            name="tickets",
            metadata={
                "description": "Historical Zendesk support tickets",
                "hnsw:space": "cosine"
            }
        )
        logger.info("Tickets collection reset complete")
    
    def reset_guides_collection(self):
        """Reset only the guides collection."""
        logger.info("Resetting guides collection...")
        try:
            self.client.delete_collection("guides")
            logger.info("Deleted guides collection")
        except Exception as e:
            logger.debug(f"No guides collection to delete: {e}")
        
        self.guides_collection = self.client.get_or_create_collection(
            name="guides",
            metadata={
                "description": "LaCuraDellAuto technical guides",
                "hnsw:space": "cosine"
            }
        )
        logger.info("Guides collection reset complete")
        
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
        
    def add_guides(self, guides_file: Optional[Path] = None, batch_size: int = 50, max_chunk_size: int = 1500):
        """Load and add guide sections to vector database.
        
        Args:
            guides_file: Path to guides JSON
            batch_size: Number of sections to process at once
            max_chunk_size: Maximum characters per chunk (for optimal embedding)
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
        seen_content = set()  # Track seen content to avoid duplicates
        
        for guide in guides:
            guide_number = guide.get('guide_number', 'UNKNOWN')
            guide_title = guide.get('title', '').strip()
            guide_url = guide.get('url', '')
            guide_description = guide.get('description', '').strip()
            
            # Process sections (use 'heading' field which matches our scraped JSON)
            for idx, section in enumerate(guide.get('sections', [])):
                # Support both 'heading' (new format) and 'title' (old format)
                section_title = section.get('heading', '') or section.get('title', '')
                section_content = section.get('content', '')
                
                # Skip empty or very short sections
                if not section_content or len(section_content) < 50:
                    continue
                
                # Skip duplicate content (deduplication)
                content_hash = hash(section_content[:200])
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # Create document with context
                if section_title:
                    document = f"Guida: {guide_title}\nSezione: {section_title}\n\n{section_content}"
                else:
                    document = f"Guida: {guide_title}\n\n{section_content}"
                
                # Chunk large sections for better embedding quality
                chunks = self._chunk_text(document, max_chunk_size)
                
                for chunk_idx, chunk in enumerate(chunks):
                    section_id = f"guide_{guide_number}_{idx}_{chunk_idx}"
                    
                    metadata = {
                        'guide_number': str(guide_number),
                        'guide_title': str(guide_title)[:500],
                        'section_title': str(section_title)[:500] if section_title else '',
                        'section_index': int(idx),
                        'chunk_index': int(chunk_idx),
                        'url': str(guide_url),
                        'content_length': len(chunk),
                        'type': 'guide_section'
                    }
                    
                    # Remove empty values
                    metadata = {k: v for k, v in metadata.items() if v not in ['', 'None', None, 0]}
                    
                    ids.append(section_id)
                    documents.append(chunk)
                    metadatas.append(metadata)
            
            # Also add tips as separate searchable documents (practical advice)
            tips = guide.get('tips', [])
            if tips:
                tips_text = f"Guida: {guide_title}\nNote e Suggerimenti:\n\n" + "\n".join(f"• {tip}" for tip in tips)
                tips_id = f"guide_{guide_number}_tips"
                
                tips_metadata = {
                    'guide_number': str(guide_number),
                    'guide_title': str(guide_title)[:500],
                    'section_title': 'Note e Suggerimenti',
                    'url': str(guide_url),
                    'content_length': len(tips_text),
                    'type': 'guide_tips'
                }
                
                ids.append(tips_id)
                documents.append(tips_text)
                metadatas.append(tips_metadata)
            
            # Add products as a searchable document
            products = guide.get('products_mentioned', [])
            if products:
                product_names = [p.get('name', '') for p in products if p.get('name') and 'ACQUISTA' not in p.get('name', '')]
                if product_names:
                    products_text = f"Guida: {guide_title}\nProdotti consigliati: {', '.join(product_names)}"
                    products_id = f"guide_{guide_number}_products"
                    
                    products_metadata = {
                        'guide_number': str(guide_number),
                        'guide_title': str(guide_title)[:500],
                        'section_title': 'Prodotti Consigliati',
                        'url': str(guide_url),
                        'type': 'guide_products'
                    }
                    
                    ids.append(products_id)
                    documents.append(products_text)
                    metadatas.append(products_metadata)
        
        # Generate embeddings and add to collection
        logger.info(f"Generating embeddings for {len(documents)} guide chunks (sections + tips + products)")
        embeddings = self.generate_embeddings(documents)
        
        logger.info(f"Adding {len(documents)} guide chunks to vector database")
        self.guides_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(documents)} guide chunks to vector database")
    
    def _chunk_text(self, text: str, max_size: int = 1500) -> List[str]:
        """Split text into chunks of approximately max_size characters.
        
        Tries to split at paragraph boundaries for better coherence.
        
        Args:
            text: Text to chunk
            max_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, split by sentences
                if len(para) > max_size:
                    sentences = para.replace('. ', '.|').split('|')
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= max_size:
                            current_chunk += (" " if current_chunk else "") + sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_size]]
    
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
    
    def add_single_ticket(self, ticket_data: Dict[str, Any]) -> bool:
        """Add a single ticket to the vector database.
        
        Args:
            ticket_data: Dictionary containing ticket information. Should have:
                - ticket_id (required)
                - subject (optional)
                - description (optional)
                - conversation (optional list of messages)
                - status (optional)
                - priority (optional)
                - created_at (optional)
                - updated_at (optional)
                - searchable_text (optional, will be generated if not provided)
        
        Returns:
            True if ticket was added successfully, False otherwise
        """
        try:
            # Ensure collections exist
            if not self.tickets_collection:
                self.create_collections()
            
            # Generate ticket ID
            ticket_id = ticket_data.get('ticket_id')
            if not ticket_id or ticket_id == '':
                # Generate a unique ID if not provided
                import time
                import random
                ticket_id = f"uploaded_{int(time.time())}_{random.randint(1000, 9999)}"
            
            ticket_id_str = f"ticket_{ticket_id}"
            
            # Check if ticket already exists
            existing = self.tickets_collection.get(ids=[ticket_id_str])
            if existing['ids']:
                logger.warning(f"Ticket {ticket_id} already exists, skipping")
                return False
            
            # Generate searchable_text if not provided
            if 'searchable_text' not in ticket_data or not ticket_data.get('searchable_text'):
                searchable_text = self._generate_searchable_text(ticket_data)
            else:
                searchable_text = ticket_data['searchable_text']
            
            if not searchable_text or len(searchable_text.strip()) < 10:
                logger.warning(f"Ticket {ticket_id} has insufficient content, skipping")
                return False
            
            # Prepare metadata
            metadata = {
                'ticket_id': str(ticket_id),
                'subject': str(ticket_data.get('subject', 'Uploaded Ticket'))[:500],
                'status': str(ticket_data.get('status', 'open')),
                'priority': str(ticket_data.get('priority', '')),
                'created_at': str(ticket_data.get('created_at', '')),
                'updated_at': str(ticket_data.get('updated_at', '')),
                'comment_count': int(ticket_data.get('comment_count', len(ticket_data.get('conversation', [])))),
                'type': 'ticket',
                'source': 'uploaded'  # Mark as uploaded
            }
            
            # Remove empty values
            metadata = {k: v for k, v in metadata.items() if v not in ['', 'None', None]}
            
            # Generate embedding
            embedding = self.generate_embeddings([searchable_text])[0]
            
            # Add to collection
            self.tickets_collection.add(
                ids=[ticket_id_str],
                documents=[searchable_text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            logger.info(f"Successfully added uploaded ticket {ticket_id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding ticket to vector database: {e}")
            return False
    
    def _generate_searchable_text(self, ticket_data: Dict[str, Any]) -> str:
        """Generate searchable text from ticket data.
        
        Args:
            ticket_data: Dictionary containing ticket information
        
        Returns:
            Formatted searchable text string
        """
        parts = []
        
        # Add subject
        subject = ticket_data.get('subject', '')
        if subject:
            parts.append(f"Subject: {subject}")
        
        # Add description
        description = ticket_data.get('description', '')
        if description:
            parts.append(f"\nDescription: {description}")
        
        # Add conversation messages
        conversation = ticket_data.get('conversation', [])
        if conversation:
            for msg in conversation:
                author = msg.get('author_name', msg.get('author_id', 'Unknown'))
                body = msg.get('plain_body', msg.get('body', ''))
                if body:
                    parts.append(f"\n{author}: {body}")
        else:
            # If no conversation, try customer_messages and agent_messages
            customer_messages = ticket_data.get('customer_messages', [])
            agent_messages = ticket_data.get('agent_messages', [])
            
            all_messages = sorted(
                customer_messages + agent_messages,
                key=lambda x: x.get('created_at', '')
            )
            
            for msg in all_messages:
                author = msg.get('author_name', msg.get('author_id', 'Unknown'))
                body = msg.get('plain_body', msg.get('body', ''))
                if body:
                    parts.append(f"\n{author}: {body}")
        
        return "\n".join(parts).strip()

