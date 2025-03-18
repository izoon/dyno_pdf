# Dyno PDF Framework

A Python-based framework for intelligent PDF processing and text extraction. The framework uses multiple PDF processing libraries and automatically selects the best method based on document characteristics.

## Features

- **Intelligent Method Selection**: Automatically chooses the best PDF processing method based on document characteristics:
  - Table-heavy documents → PDFPlumber
  - Scanned documents → Tesseract OCR
  - Image-heavy documents → Tesseract OCR
  - Digital text documents → PyMuPDF

- **Multiple Processing Methods**:
  - PDFPlumber: Best for table-heavy documents
  - Tesseract OCR: Best for scanned and image-heavy documents
  - PyMuPDF: General-purpose PDF processing

- **Smart Analysis**:
  - Text density analysis
  - Image density analysis
  - Table detection
  - Scanned document detection

- **Performance Tracking**:
  - Processing time tracking
  - Success rate monitoring
  - Method effectiveness metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dyno_pdf.git
cd dyno_pdf
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from pdf_agent.agent import PDFProcessingDecisionAgent

# Initialize the agent
agent = PDFProcessingDecisionAgent()

# Process a single PDF
result = agent.process_pdf("path/to/your.pdf")
print(result)
```

### Batch Processing

```python
from pdf_agent.agent import PDFProcessingDecisionAgent

# Initialize the agent
agent = PDFProcessingDecisionAgent()

# Process multiple PDFs
results = agent.process_pdfs(["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"])
for result in results:
    print(result)
```

## Testing

The framework includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pdf_agent.py -v
```

Current test results:
- ✅ Agent initialization
- ✅ Document characteristics analysis
- ✅ Method selection logic
- ✅ Text extraction methods
- ✅ Database operations
- ✅ Error handling
- ✅ Performance metrics
- ✅ Batch processing

## Project Structure

```
dyno_pdf/
├── pdf_agent/
│   ├── __init__.py
│   ├── agent.py           # Main agent implementation
│   ├── extractors.py      # Text extraction implementations
│   └── database.py        # Database operations
├── tests/
│   ├── test_pdf_agent.py
│   ├── test_pdf_processing.py
│   └── test_pdfs/        # Test PDF files
├── setup.py
└── README.md
```

## Method Selection Logic

The framework analyzes PDF characteristics to select the best processing method:

1. **Table Detection**:
   - Very high table indicators (>10) → PDFPlumber
   - High table indicators (>5) → PDFPlumber with reduced priority

2. **Scanned Document Detection**:
   - High scanned indicators → Tesseract OCR
   - Medium scanned indicators → Tesseract OCR with reduced priority

3. **Image Density**:
   - High image density → Tesseract OCR
   - Medium image density → Tesseract OCR with reduced priority

4. **Text Density**:
   - High text density → PyMuPDF
   - Medium text density → PyMuPDF with reduced priority

## Performance

The framework has been tested with various PDF types:
- Digital text documents
- Scanned documents
- Table-heavy documents
- Image-heavy documents
- Mixed-content documents

Current OCR confidence scores range from 20-28%, indicating potential for improvement in text extraction quality.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 