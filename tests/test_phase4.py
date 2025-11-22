"""Tests for Phase 4: RAG Pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.phase4.vector_db import VectorDBManager
from src.phase4.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer model."""
    with patch('src.phase4.vector_db.SentenceTransformer') as mock:
        model_instance = Mock()
        model_instance.get_sentence_embedding_dimension.return_value = 384
        model_instance.encode.return_value = [[0.1] * 384]  # Mock embedding
        mock.return_value = model_instance
        yield mock


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    with patch('src.phase4.vector_db.chromadb.PersistentClient') as mock:
        client_instance = Mock()
        
        # Mock collections
        collection_mock = Mock()
        collection_mock.count.return_value = 0
        collection_mock.add = Mock()
        collection_mock.query.return_value = {
            'ids': [['test_id']],
            'documents': [['test document']],
            'metadatas': [[{'type': 'test'}]],
            'distances': [[0.5]]
        }
        
        client_instance.get_or_create_collection.return_value = collection_mock
        client_instance.delete_collection = Mock()
        mock.return_value = client_instance
        
        yield mock


class TestVectorDBManager:
    """Test Vector Database Manager."""
    
    def test_init(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test initialization."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        
        assert db.db_path == tmp_path / "test_db"
        assert db.embedding_dim == 384
        mock_embedding_model.assert_called_once()
    
    def test_create_collections(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test collection creation."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        db.create_collections()
        
        assert db.tickets_collection is not None
        assert db.guides_collection is not None
    
    def test_generate_embeddings(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test embedding generation."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        texts = ["test text 1", "test text 2"]
        
        embeddings = db.generate_embeddings(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) > 0
    
    def test_search_tickets(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test ticket search."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        db.create_collections()
        
        results = db.search_tickets("test query", n_results=3)
        
        assert 'ids' in results
        assert 'documents' in results
        assert 'metadatas' in results
    
    def test_search_guides(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test guide search."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        db.create_collections()
        
        results = db.search_guides("test query", n_results=3)
        
        assert 'ids' in results
        assert 'documents' in results
        assert 'metadatas' in results
    
    def test_get_stats(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test statistics retrieval."""
        db = VectorDBManager(db_path=tmp_path / "test_db")
        db.create_collections()
        
        stats = db.get_stats()
        
        assert 'tickets' in stats
        assert 'guides' in stats
        assert isinstance(stats['tickets'], int)
        assert isinstance(stats['guides'], int)


class TestRAGPipeline:
    """Test RAG Pipeline."""
    
    def test_init(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test RAG pipeline initialization."""
        db_manager = VectorDBManager(db_path=tmp_path / "test_db")
        pipeline = RAGPipeline(db_manager=db_manager, model="test-model")
        
        assert pipeline.model == "test-model"
        assert pipeline.db_manager is not None
    
    def test_format_context(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test context formatting."""
        db_manager = VectorDBManager(db_path=tmp_path / "test_db")
        pipeline = RAGPipeline(db_manager=db_manager)
        
        results = {
            'tickets': {
                'documents': [['ticket content']],
                'metadatas': [[{'subject': 'Test Subject', 'status': 'solved'}]]
            },
            'guides': {
                'documents': [['guide content']],
                'metadatas': [[{'guide_title': 'Test Guide', 'section_title': 'Test Section'}]]
            }
        }
        
        context = pipeline.format_context(results)
        
        assert 'HISTORICAL TICKETS' in context
        assert 'TECHNICAL GUIDES' in context
        assert 'Test Subject' in context
        assert 'Test Guide' in context
    
    def test_create_prompt(self, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test prompt creation."""
        db_manager = VectorDBManager(db_path=tmp_path / "test_db")
        pipeline = RAGPipeline(db_manager=db_manager)
        
        query = "How do I wash my car?"
        context = "Context information here"
        
        prompt = pipeline.create_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
        assert 'LaCuraDellAuto' in prompt
    
    @patch('src.phase4.rag_pipeline.ollama.generate')
    def test_generate_response(self, mock_ollama, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test response generation."""
        mock_ollama.return_value = {'response': 'Generated response'}
        
        db_manager = VectorDBManager(db_path=tmp_path / "test_db")
        pipeline = RAGPipeline(db_manager=db_manager)
        
        response = pipeline.generate_response("test prompt")
        
        assert response == 'Generated response'
        mock_ollama.assert_called_once()
    
    @patch('src.phase4.rag_pipeline.ollama.list')
    def test_check_ollama_status(self, mock_list, mock_chroma_client, mock_embedding_model, tmp_path):
        """Test Ollama status check."""
        mock_list.return_value = {'models': [{'name': 'mistral:latest'}]}
        
        db_manager = VectorDBManager(db_path=tmp_path / "test_db")
        pipeline = RAGPipeline(db_manager=db_manager, model="mistral")
        
        status = pipeline.check_ollama_status()
        
        assert status['ollama_running'] is True
        assert 'available_models' in status

