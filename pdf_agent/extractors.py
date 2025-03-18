import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PDFExtractor(ABC):
    """Base class for PDF extractors."""
    
    @abstractmethod
    def extract_text(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract text from PDF."""
        pass

class PyMuPDFExtractor(PDFExtractor):
    """PDF text extractor using PyMuPDF."""
    
    def extract_text(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = {}
            metadata = {}
            
            # Get metadata
            metadata = doc.metadata
            
            # Process specified pages or all pages
            pages = page_numbers if page_numbers else range(len(doc))
            for page_num in pages:
                page = doc[page_num]
                text[page_num + 1] = page.get_text()
            
            doc.close()
            return {"text": text, "metadata": metadata}
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {str(e)}")
            raise

class PDFPlumberExtractor(PDFExtractor):
    """PDF text extractor using PDFPlumber."""
    
    def extract_text(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract text from PDF using PDFPlumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = {}
                metadata = {}
                
                # Get metadata
                metadata = pdf.metadata
                
                # Process specified pages or all pages
                pages = page_numbers if page_numbers else range(len(pdf.pages))
                for page_num in pages:
                    page = pdf.pages[page_num]
                    text[page_num + 1] = page.extract_text()
                
                return {"text": text, "metadata": metadata}
        except Exception as e:
            logger.error(f"PDFPlumber extraction error: {str(e)}")
            raise

class TesseractExtractor(PDFExtractor):
    """PDF text extractor using Tesseract OCR."""
    
    def extract_text(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract text from PDF using Tesseract OCR."""
        try:
            doc = fitz.open(pdf_path)
            text = {}
            metadata = {}
            ocr_confidence = {}
            
            # Get metadata
            metadata = doc.metadata
            
            # Process specified pages or all pages
            pages = page_numbers if page_numbers else range(len(doc))
            for page_num in pages:
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR
                ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Extract text and confidence
                page_text = ""
                confidences = []
                for i in range(len(ocr_result['text'])):
                    if float(ocr_result['conf'][i]) > 0:
                        page_text += ocr_result['text'][i] + " "
                        confidences.append(float(ocr_result['conf'][i]))
                
                text[page_num + 1] = page_text.strip()
                ocr_confidence[page_num + 1] = sum(confidences) / len(confidences) if confidences else 0
            
            doc.close()
            return {
                "text": text,
                "metadata": metadata,
                "ocr_confidence": ocr_confidence
            }
        except Exception as e:
            logger.error(f"Tesseract extraction error: {str(e)}")
            raise

# Helper functions
def analyze_pdf_characteristics(pdf_path: str) -> Dict[str, float]:
    """Analyze PDF characteristics to help in method selection."""
    try:
        doc = fitz.open(pdf_path)
        
        # Initialize counters
        total_pages = len(doc)
        total_text_blocks = 0
        total_image_blocks = 0
        total_table_blocks = 0
        total_scanned_blocks = 0
        
        # Analyze each page
        for page in doc:
            # Get text blocks
            text_blocks = page.get_text("blocks")
            total_text_blocks += len(text_blocks)
            
            # Get images
            image_list = page.get_images()
            total_image_blocks += len(image_list)
            
            # Check for tables (simple heuristic)
            table_blocks = [block for block in text_blocks if len(block[4].split('\n')) > 3]
            total_table_blocks += len(table_blocks)
            
            # Check for scanned content (heuristic based on text density)
            text_density = len(page.get_text()) / (page.rect.width * page.rect.height)
            if text_density < 0.1:  # Low text density might indicate scanned content
                total_scanned_blocks += 1
        
        doc.close()
        
        # Calculate densities
        total_blocks = total_text_blocks + total_image_blocks
        text_density = total_text_blocks / total_blocks if total_blocks > 0 else 0
        image_density = total_image_blocks / total_blocks if total_blocks > 0 else 0
        table_density = total_table_blocks / total_pages if total_pages > 0 else 0
        scanned_density = total_scanned_blocks / total_pages if total_pages > 0 else 0
        
        return {
            "text_density": text_density,
            "image_density": image_density,
            "table_indicators": table_density,
            "scanned_indicators": scanned_density
        }
    except Exception as e:
        logger.error(f"PDF analysis error: {str(e)}")
        raise

def determine_document_type(characteristics: Dict[str, float]) -> str:
    """Determine document type based on characteristics."""
    if characteristics["scanned_indicators"] > 0.3:
        return "scanned"
    elif characteristics["table_indicators"] > 0.2:
        return "table_heavy"
    else:
        return "digital_text"

def calculate_non_ascii_ratio(text: str) -> float:
    """Calculate the ratio of non-ASCII characters in text."""
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) if len(text) > 0 else 0.0 