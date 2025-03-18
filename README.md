# Dyno Framework

A framework for PDF processing and other document workflows, built on top of dynoagent.

## Features

- PDF Processing Library
  - Multiple extraction methods (PyMuPDF, PDFPlumber, Tesseract)
  - Intelligent method selection with reinforcement learning
  - Quality assessment and validation
  - SQLite storage for results
  - Integration with LlamaIndex for document processing
  - Support for input dependencies and preprocessing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from pdf_agent.agent import PDFProcessingDecisionAgent

# Initialize the agent
agent = PDFProcessingDecisionAgent(
    name="pdf_processor",
    sampling_pages=2,
    enable_learning=True
)

# Process a PDF
result = agent.process_pdf("path/to/pdf")
```

## Project Structure

```
dyno_pdf/
├── pdf_agent/              # Main implementation directory
│   ├── agent.py           # Core PDF processing agent
│   ├── dyno_llamaindex.py # LlamaIndex integration
│   ├── extractors.py      # PDF text extraction implementations
│   ├── test_pdf_agent.py  # Tests
│   └── test_pdfs/         # Test PDF files
├── requirements.txt       # Project dependencies
├── setup.py              # Package configuration
└── pdf_extraction.db     # SQLite database for results
```

## Dependencies

- PyMuPDF (fitz) - PDF processing
- PDFPlumber - PDF text extraction
- Tesseract - OCR capabilities
- OpenCV & Pillow - Image processing
- LlamaIndex - Document processing and indexing
- dynoagent - Core agent framework

## Development

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest pdf_agent/test_pdf_agent.py`

## License

MIT License 