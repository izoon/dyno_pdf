import os
import pytest
from pdf_agent.agent import PDFProcessingDecisionAgent
from pdf_agent.extractors import PyMuPDFExtractor, PDFPlumberExtractor, TesseractExtractor

@pytest.fixture
def pdf_agent():
    """Fixture to create a PDFProcessingDecisionAgent instance."""
    agent = PDFProcessingDecisionAgent(
        name="test_pdf_processor",
        skills=["pdf_extraction", "ocr", "document_analysis"],
        goal="Test PDF processing methods",
        sampling_pages=2
    )
    return agent

@pytest.fixture
def test_pdfs_dir():
    """Fixture to provide the test PDFs directory path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "test_pdfs")

def test_agent_initialization(pdf_agent):
    """Test agent initialization and attributes."""
    assert pdf_agent.name == "test_pdf_processor"
    assert "pdf_extraction" in pdf_agent.skills
    assert pdf_agent.sampling_pages == 2
    assert hasattr(pdf_agent, "process_pdf")

def test_document_characteristics(pdf_agent, test_pdfs_dir):
    """Test document characteristics extraction."""
    # Test with a table-heavy PDF
    aral_pdf = os.path.join(test_pdfs_dir, "aral_20240926.pdf")
    characteristics = pdf_agent._analyze_document_characteristics(aral_pdf)
    
    assert isinstance(characteristics, dict)
    assert "page_count" in characteristics
    assert "text_density" in characteristics
    assert "table_indicators" in characteristics
    assert characteristics["table_indicators"] > 0  # Should detect tables

def test_method_selection(pdf_agent, test_pdfs_dir):
    """Test method selection logic for different types of PDFs."""
    test_cases = [
        ("aral_20240926.pdf", "pdfplumber"),  # Table-heavy document
        ("happyitaly_20240306_001.pdf", "tesseract"),  # Image-heavy document
    ]
    
    for filename, expected_method in test_cases:
        pdf_path = os.path.join(test_pdfs_dir, filename)
        method, score = pdf_agent._select_processing_method(pdf_path)
        assert method == expected_method, f"Expected {expected_method} for {filename}, got {method}"

def test_text_extraction_methods():
    """Test individual extraction methods."""
    # Test PyMuPDF extractor
    pymupdf = PyMuPDFExtractor()
    assert hasattr(pymupdf, "extract_text")
    
    # Test PDFPlumber extractor
    pdfplumber = PDFPlumberExtractor()
    assert hasattr(pdfplumber, "extract_text")
    
    # Test Tesseract extractor
    tesseract = TesseractExtractor()
    assert hasattr(tesseract, "extract_text")

def test_database_operations(pdf_agent, test_pdfs_dir):
    """Test database operations."""
    # Process a PDF
    pdf_path = os.path.join(test_pdfs_dir, "aral_20240926.pdf")
    documents = pdf_agent.process_pdf(pdf_path)
    
    # Test querying results
    results = pdf_agent.query_results(filename="aral_20240926.pdf")
    assert isinstance(results, list)
    
    # Test result contents
    if results:
        result = results[0]
        assert "document_type" in result
        assert "method_used" in result
        assert "filename" in result
        assert result["filename"] == "aral_20240926.pdf"

def test_error_handling(pdf_agent):
    """Test error handling for various scenarios."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        pdf_agent.process_pdf("nonexistent.pdf")
    
    # Test with invalid file type
    with pytest.raises(ValueError):
        pdf_agent.process_pdf("test.txt")

def test_performance_metrics(pdf_agent, test_pdfs_dir):
    """Test performance metrics collection."""
    pdf_path = os.path.join(test_pdfs_dir, "aral_20240926.pdf")
    
    # Process PDF and check metrics
    documents = pdf_agent.process_pdf(pdf_path)
    metrics = pdf_agent.get_performance_metrics(pdf_path)
    
    assert isinstance(metrics, dict)
    assert "processing_time" in metrics
    assert "method_used" in metrics
    assert "success" in metrics

def test_batch_processing(pdf_agent, test_pdfs_dir):
    """Test batch processing of multiple PDFs."""
    pdf_files = [f for f in os.listdir(test_pdfs_dir) if f.endswith(".pdf")]
    results = []
    
    for pdf_file in pdf_files[:2]:  # Test with first 2 PDFs
        pdf_path = os.path.join(test_pdfs_dir, pdf_file)
        result = pdf_agent.process_pdf(pdf_path)
        results.append(result)
    
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)  # Each result should be a list of pages 