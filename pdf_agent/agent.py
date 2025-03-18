from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import os
import io
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import cv2
from dynoagent.dyno_agent_with_tools import DynoAgentWithTools
from .dyno_llamaindex import DynoDataLoader
from llama_index.core import Document
import logging
import json
import sqlite3
from datetime import datetime
from .extractors import PyMuPDFExtractor, PDFPlumberExtractor, TesseractExtractor

class PDFProcessingDecisionAgent(DynoAgentWithTools):
    """
    Decision agent that intelligently selects the best PDF processing method based on document characteristics.
    Uses reinforcement learning principles to improve selection over time.
    """
    
    def __init__(self, name, role="PDF Processor", skills=None, goal="Select optimal PDF processing method",
                 enable_learning=True, learning_threshold=10, accuracy_boost_factor=1.5,
                 use_rl_decision_agent=True, input_dependencies=None, tools_dataloaders=None,
                 llm_provider=None, temperature=0.7, max_tokens=1500, sampling_pages=3):
        """
        Initialize the PDF Processing Decision Agent.
        
        Args:
            name: Agent name
            role: Agent role (default: "PDF Processor")
            skills: Agent skills list
            goal: Agent goal
            enable_learning: Whether to enable learning from past decisions
            learning_threshold: Number of inputs needed before adjusting sequencing
            accuracy_boost_factor: Factor to suggest higher input data for accuracy improvement
            use_rl_decision_agent: Whether to use RL Decision Agent
            input_dependencies: List of input dependencies for processing
            tools_dataloaders: Dictionary of additional tools or data loaders
            llm_provider: LLM provider to use
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens for LLM response
            sampling_pages: Number of pages to sample for analysis
        """
        if skills is None:
            skills = ["PDF Analysis", "Method Selection", "OCR Evaluation"]
            
        super().__init__(
            name=name,
            role=role,
            skills=skills,
            goal=goal,
            enable_learning=enable_learning,
            learning_threshold=learning_threshold,
            accuracy_boost_factor=accuracy_boost_factor,
            use_rl_decision_agent=use_rl_decision_agent,
            input_dependencies=input_dependencies,
            tools_dataloaders=tools_dataloaders,
            llm_provider=llm_provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.sampling_pages = sampling_pages
        self.available_methods = ["pymupdf", "pdfplumber", "tesseract"]
        
        # Initialize extractors
        self.extractors = {
            "pymupdf": PyMuPDFExtractor(),
            "pdfplumber": PDFPlumberExtractor(),
            "tesseract": TesseractExtractor()
        }
        
        # Method performance scores by document type
        self.method_scores = {
            "digital_text": {"pymupdf": 100, "pdfplumber": 80, "tesseract": 30},
            "scanned": {"pymupdf": 20, "pdfplumber": 30, "tesseract": 90},
            "table_heavy": {"pymupdf": 40, "pdfplumber": 90, "tesseract": 30},
            "image_heavy": {"pymupdf": 30, "pdfplumber": 30, "tesseract": 90},
            "mixed": {"pymupdf": 50, "pdfplumber": 50, "tesseract": 70}
        }
        
        # Method performance characteristics
        self.method_performance = {
            "pymupdf": {"speed": 90, "accuracy": 80, "table_quality": 60, "image_handling": 70},
            "pdfplumber": {"speed": 70, "accuracy": 80, "table_quality": 90, "image_handling": 60},
            "tesseract": {"speed": 50, "accuracy": 70, "table_quality": 50, "image_handling": 90}
        }
        
        # Thresholds for document characteristics
        self.characteristic_thresholds = {
            "text_density": {
                "low": 0.35,
                "medium": 0.7,
                "high": 1.2
            },
            "image_density": {
                "low": 0.05,
                "medium": 0.1,
                "high": 0.2
            },
            "table_indicators": {
                "low": 1,
                "medium": 3,
                "high": 5
            },
            "scanned_indicators": {
                "low": 0.2,
                "medium": 0.4,
                "high": 0.6
            }
        }
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Add file handler if not already added
        if not self.logger.handlers:
            fh = logging.FileHandler('pdf_processing.log')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            # Add console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # Initialize database
        self.init_database()
        
        # Decision history for learning
        self.decision_history = []
        
        # Register default tools
        self._register_default_tools()
    
    def init_database(self):
        """Initialize SQLite database for storing extraction results."""
        conn = sqlite3.connect('pdf_extraction.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS pdf_extractions
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             filename TEXT NOT NULL,
             document_type TEXT,
             method_used TEXT,
             rationale TEXT,
             characteristics JSON,
             extracted_text JSON,
             metadata JSON,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        conn.commit()
        conn.close()
    
    def _register_default_tools(self):
        """Register default tools and data loaders."""
        # Register extraction methods as tools
        self.tools_dataloaders = {
            "extract_with_pymupdf": self._extract_with_pymupdf,
            "extract_with_pdfplumber": self._extract_with_pdfplumber,
            "extract_with_tesseract": self._extract_with_tesseract
        }
    
    def register_tool(self, name: str, tool_function: Callable) -> None:
        """
        Register a new tool or data loader.
        
        Args:
            name: Name of the tool
            tool_function: Function implementing the tool
        """
        self.tools_dataloaders[name] = tool_function
        self.history.append({
            "task": "Register tool",
            "context": f"Added new tool: {name}",
            "role": self.role
        })
    
    def add_input_dependency(self, dependency: Any) -> None:
        """
        Add a new input dependency.
        
        Args:
            dependency: Input dependency to add
        """
        self.input_dependencies.append(dependency)
        self.history.append({
            "task": "Add input dependency",
            "context": f"Added new dependency: {type(dependency).__name__}",
            "role": self.role
        })
    
    def get_available_tools(self) -> List[str]:
        """Get names of all available tools and data loaders."""
        return list(self.tools_dataloaders.keys())
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Use a registered tool.
        
        Args:
            tool_name: Name of the tool to use
            *args, **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name not in self.tools_dataloaders:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {self.get_available_tools()}")
            
        tool = self.tools_dataloaders[tool_name]
        
        self.history.append({
            "task": "Use tool",
            "context": f"Used tool: {tool_name}",
            "role": self.role
        })
        
        return tool(*args, **kwargs)
    
    def _analyze_document_characteristics(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF characteristics to determine document type.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document characteristics
        """
        self.history.append({
            "task": "Analyze PDF characteristics",
            "context": pdf_path,
            "role": self.role
        })
        
        try:
            # Open PDF with PyMuPDF for initial analysis
            doc = None
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                self.logger.error(f"Failed to open PDF: {pdf_path} - {str(e)}")
                raise FileNotFoundError(f"Failed to open PDF: {str(e)}")
            
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
                "page_count": total_pages,
                "text_density": text_density,
                "image_density": image_density,
                "table_indicators": table_density,
                "scanned_indicators": scanned_density
            }
        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"PDF analysis error: {str(e)}")
            raise

    def _select_processing_method(self, pdf_path: str) -> Tuple[str, float]:
        """
        Select the best processing method based on document characteristics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (selected_method, confidence_score)
        """
        self.logger.info(f"Selecting method for {pdf_path}")
        
        try:
            # Get document characteristics
            characteristics = self._analyze_document_characteristics(pdf_path)
            doc_type = self._determine_document_type(characteristics)
            
            self.logger.info(f"Document type: {doc_type}")
            self.logger.info(f"Characteristics: {characteristics}")
            
            # Get initial scores based on document type
            scores = self.method_scores[doc_type].copy()
            self.logger.info(f"Initial scores: {scores}")
            
            # First check for very high table indicators (highest priority)
            if characteristics["table_indicators"] > self.characteristic_thresholds["table_indicators"]["high"] * 2:
                self.logger.info("Applied table adjustment (very high)")
                scores["pdfplumber"] += 100
                scores["pymupdf"] -= 40
                scores["tesseract"] -= 40
            # Then check for scanned documents
            elif characteristics["scanned_indicators"] > self.characteristic_thresholds["scanned_indicators"]["high"]:
                self.logger.info("Applied scanned document adjustment (high)")
                scores["tesseract"] += 100
                scores["pymupdf"] -= 30
                scores["pdfplumber"] -= 30
            elif characteristics["scanned_indicators"] > self.characteristic_thresholds["scanned_indicators"]["medium"]:
                self.logger.info("Applied scanned document adjustment (medium)")
                scores["tesseract"] += 60
                scores["pymupdf"] -= 20
                scores["pdfplumber"] -= 20
            
            # Then check for image density
            if characteristics["image_density"] > self.characteristic_thresholds["image_density"]["high"]:
                self.logger.info("Applied image density adjustment (high)")
                scores["tesseract"] += 100
                scores["pymupdf"] -= 30
                scores["pdfplumber"] -= 30
            elif characteristics["image_density"] > self.characteristic_thresholds["image_density"]["medium"]:
                self.logger.info("Applied image density adjustment (medium)")
                scores["tesseract"] += 60
                scores["pymupdf"] -= 20
                scores["pdfplumber"] -= 20
            
            # Only apply table adjustments if not already handled and neither scanned nor image-heavy
            if (characteristics["table_indicators"] <= self.characteristic_thresholds["table_indicators"]["high"] * 2 and
                characteristics["scanned_indicators"] <= self.characteristic_thresholds["scanned_indicators"]["medium"] and
                characteristics["image_density"] <= self.characteristic_thresholds["image_density"]["medium"]):
                if characteristics["table_indicators"] > self.characteristic_thresholds["table_indicators"]["high"]:
                    self.logger.info("Applied table adjustment (high)")
                    scores["pdfplumber"] += 50
                    scores["pymupdf"] -= 40
                    scores["tesseract"] -= 20
                elif characteristics["table_indicators"] > self.characteristic_thresholds["table_indicators"]["medium"]:
                    self.logger.info("Applied table adjustment (medium)")
                    scores["pdfplumber"] += 30
                    scores["pymupdf"] -= 20
                    scores["tesseract"] -= 10
            
            # Only apply text density adjustments if neither scanned nor image-heavy
            if (characteristics["scanned_indicators"] <= self.characteristic_thresholds["scanned_indicators"]["medium"] and
                characteristics["image_density"] <= self.characteristic_thresholds["image_density"]["medium"]):
                if characteristics["text_density"] > self.characteristic_thresholds["text_density"]["high"]:
                    self.logger.info("Applied text density adjustment (high)")
                    scores["pymupdf"] += 30
                    scores["tesseract"] -= 20
                elif characteristics["text_density"] > self.characteristic_thresholds["text_density"]["medium"]:
                    self.logger.info("Applied text density adjustment (medium)")
                    scores["pymupdf"] += 20
                    scores["tesseract"] -= 10
            
            # Ensure no negative scores
            scores = {k: max(0, v) for k, v in scores.items()}
            
            self.logger.info(f"Final scores: {scores}")
            
            # Select method with highest score
            selected_method = max(scores.items(), key=lambda x: x[1])
            self.logger.info(f"Selected method: {selected_method[0]} with score {selected_method[1]}")
            
            # Store the selected method for metrics
            self._last_method_used = selected_method[0]
            self._last_success = True  # Default to True, will be updated if extraction fails
            
            return selected_method[0], selected_method[1]
        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.warning(f"Error in analysis: {str(e)}")
            # Default to PyMuPDF if analysis fails
            self._last_method_used = "pymupdf"
            self._last_success = False
            return "pymupdf", 100

    def process_pdf(self, pdf_path: str, extraction_priority: Optional[str] = None, 
                    use_dependencies: bool = True, custom_method: Optional[str] = None) -> List[Document]:
        """
        Process the PDF using the selected method and convert to LlamaIndex documents.
        If the selected method fails or produces low-quality output, try the next best method.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_priority: Priority type ("speed", "accuracy", "tables", "images")
            use_dependencies: Whether to use registered input dependencies
            custom_method: Force using a specific method instead of auto-selecting
            
        Returns:
            List of LlamaIndex Document objects with quality status and QA flags
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File {pdf_path} is not a PDF file")
        
        # Record start time
        start_time = datetime.now()
        
        # Select method (or use custom method if specified)
        if custom_method:
            if custom_method not in self.available_methods:
                raise ValueError(f"Custom method '{custom_method}' not in available methods: {self.available_methods}")
            method = custom_method
            method_score = 100
        else:
            method, method_score = self._select_processing_method(pdf_path)
        
        # Get ordered list of methods by score for fallback
        ordered_methods = [(method, method_score)]
        if not custom_method:
            # Get other methods sorted by score
            doc_type = self._determine_document_type(self._analyze_document_characteristics(pdf_path))
            scores = self.method_scores[doc_type].copy()
            ordered_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Record processing time
        self._last_processing_time = (datetime.now() - start_time).total_seconds()
        
        for current_method, score in ordered_methods:
            self.logger.info(f"Trying method {current_method} with score {score}")
            
            try:
                # Apply input dependencies if available and requested
                preprocessed_path = pdf_path
                if use_dependencies and self.input_dependencies:
                    preprocessed_path = self._apply_dependencies(pdf_path)
                
                # Try the current method
                extracted_text = self.use_tool(f"extract_with_{current_method}", preprocessed_path)
                
                # Check if extraction was successful
                if not extracted_text or all(not text.strip() for text in extracted_text):
                    self.logger.warning(f"No text could be extracted from {pdf_path} using {current_method}")
                    continue
                
                # For Tesseract, get OCR confidence scores
                if current_method == "tesseract":
                    ocr_scores = self._get_ocr_confidence_scores(preprocessed_path)
                    self.logger.info(f"OCR confidence scores: {ocr_scores}")
                
                # Process each page
                documents = []
                for i, page_text in enumerate(extracted_text):
                    # Initialize quality issues for this page
                    page_quality_issues = []
                    
                    # Basic quality checks
                    if len(page_text.strip()) < 50:  # Minimum text length
                        page_quality_issues.append("Insufficient text length")
                    
                    # Check for common indicators of low quality
                    words = page_text.split()
                    if len(words) < 10:  # Minimum word count
                        page_quality_issues.append("Insufficient word count")
                    
                    # Check for poor word spacing or formatting
                    if any(len(word) > 50 for word in words):  # Unusually long words
                        page_quality_issues.append("Poor word spacing")
                    
                    # Check for lack of line breaks
                    if page_text.count('\n') < 3:  # Minimum line breaks
                        page_quality_issues.append("Poor text formatting")
                    
                    # Check for character encoding issues
                    if any(ord(c) > 127 for c in page_text):  # Non-ASCII characters
                        page_quality_issues.append("Character encoding issues")
                    
                    # Determine quality status
                    quality_status = "good"
                    if page_quality_issues:
                        quality_status = "low_quality"
                    
                    # Create document with quality metadata
                    doc = Document(
                        text=page_text,
                        metadata={
                            "source": pdf_path,
                            "page_number": i + 1,
                            "extraction_method": current_method,
                            "quality_status": quality_status,
                            "qa_required": bool(page_quality_issues),
                            "quality_issues": page_quality_issues,
                            "document_type": doc_type,
                            "method_score": score,
                            "ocr_confidence": ocr_scores[i] if current_method == "tesseract" else None
                        }
                    )
                    documents.append(doc)
                
                # If we got here, the method worked well enough
                self.logger.info(f"Successfully processed PDF with {current_method}")
                
                # Update success rating for the last decision
                if self.decision_history and not custom_method:
                    self.decision_history[-1]["success_rating"] = 1.0
                
                return documents
                
            except Exception as e:
                self.logger.warning(f"Error processing PDF with {current_method}: {str(e)}")
                continue
        
        # If we get here, all methods failed
        self.logger.error(f"All methods failed to process {pdf_path}")
        
        # Update success rating for the last decision
        if self.decision_history and not custom_method:
            self.decision_history[-1]["success_rating"] = 0.0
        
        # Return a document with error status for QA review
        return [Document(
            text="",
            metadata={
                "source": pdf_path,
                "page_number": 1,
                "extraction_method": "all_failed",
                "quality_status": "error",
                "qa_required": True,
                "error_message": "All extraction methods failed",
                "qa_queue": "manual_review",
                "document_type": doc_type
            }
        )]
    
    def _apply_dependencies(self, pdf_path: str) -> str:
        """
        Apply input dependencies to preprocess the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the preprocessed PDF (may be the same as input)
        """
        # If no dependencies, return the original path
        if not self.input_dependencies:
            return pdf_path
            
        current_path = pdf_path
        
        # Apply each dependency in sequence
        for dependency in self.input_dependencies:
            # Check if dependency has a process method
            if hasattr(dependency, 'process') and callable(getattr(dependency, 'process')):
                try:
                    # Dependency should return a path to the processed file
                    result = dependency.process(current_path)
                    if result and isinstance(result, str):
                        current_path = result
                except Exception as e:
                    print(f"Error applying dependency {type(dependency).__name__}: {str(e)}")
            
        return current_path
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[str]:
        """Extract text using PyMuPDF."""
        extractor = self.extractors["pymupdf"]
        result = extractor.extract_text(pdf_path)
        return [text for _, text in result["text"].items()]
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[str]:
        """Extract text using PDFPlumber."""
        extractor = self.extractors["pdfplumber"]
        result = extractor.extract_text(pdf_path)
        return [text for _, text in result["text"].items()]
    
    def _extract_with_tesseract(self, pdf_path: str) -> List[str]:
        """Extract text using Tesseract OCR."""
        extractor = self.extractors["tesseract"]
        result = extractor.extract_text(pdf_path)
        return [text for _, text in result["text"].items()]
    
    def _get_ocr_confidence_scores(self, pdf_path: str) -> List[float]:
        """
        Get OCR confidence scores for each page using Tesseract.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of confidence scores (0-100) for each page
        """
        confidence_scores = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render page to image
                pix = page.get_pixmap(alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert PIL image to OpenCV format
                img_cv = np.array(img)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                
                # Preprocess image for better OCR
                img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Get OCR confidence score
                data = pytesseract.image_to_data(img_thresh, output_type=pytesseract.Output.DICT)
                if data['conf']:
                    # Average confidence score for the page
                    avg_conf = sum(float(conf) for conf in data['conf'] if conf != '-1') / len(data['conf'])
                    confidence_scores.append(avg_conf)
                else:
                    confidence_scores.append(0.0)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error getting OCR confidence scores: {str(e)}")
            confidence_scores = [0.0] * len(doc)
        
        return confidence_scores
    
    def update_learning(self, pdf_path: str, method: str, success_rating: float) -> None:
        """
        Update learning based on success or failure of a method.
        
        Args:
            pdf_path: Path to the PDF file that was processed
            method: Method that was used
            success_rating: Rating of success (0.0 to 1.0)
        """
        # Find the decision in history
        for decision in self.decision_history:
            if decision["pdf_path"] == pdf_path and decision["selected_method"] == method:
                decision["success_rating"] = success_rating
                
                # Get document type
                doc_type = decision["document_type"]
                
                # Update scores based on success rating
                learning_rate = 10  # Increased for more noticeable changes
                current_score = self.method_scores[doc_type][method]
                
                # If success, increase score; if failure, decrease score
                if success_rating > 0.5:
                    # Increase score, but not above 200
                    new_score = min(200, current_score + learning_rate * success_rating)
                else:
                    # Decrease score, but not below 0
                    new_score = max(0, current_score - learning_rate * (1 - success_rating))
                
                # Update score
                self.method_scores[doc_type][method] = new_score
                
                self.logger.info(f"Updated score for {method} on {doc_type} documents: {new_score}")
                break
    
    def get_performance_metrics(self, pdf_path: str) -> Dict[str, Any]:
        """Get performance metrics for learning analysis."""
        method_success = {method: [] for method in self.available_methods}
        doc_type_distribution = {}
        doc_type_success = {doc_type: {method: [] for method in self.available_methods} 
                          for doc_type in self.method_scores.keys()}
        
        # Get results for specific PDF if provided
        results = self.query_results(filename=os.path.basename(pdf_path))
        if results:
            result = results[0]
            method = result["method_used"]
            doc_type = result["document_type"]
            method_success[method].append(1.0)  # Assume success if we have results
            doc_type_success[doc_type][method].append(1.0)
            doc_type_distribution[doc_type] = doc_type_distribution.get(doc_type, 0) + 1
        
        # Calculate average success rate per method
        avg_success = {}
        for method, ratings in method_success.items():
            if ratings:
                avg_success[method] = sum(ratings) / len(ratings)
            else:
                avg_success[method] = 0.0
        
        # Calculate average success rate per method per document type
        doc_type_avg_success = {}
        for doc_type in doc_type_success:
            doc_type_avg_success[doc_type] = {}
            for method in doc_type_success[doc_type]:
                ratings = doc_type_success[doc_type][method]
                if ratings:
                    doc_type_avg_success[doc_type][method] = sum(ratings) / len(ratings)
                else:
                    doc_type_avg_success[doc_type][method] = 0.0
        
        return {
            "average_success_by_method": avg_success,
            "document_type_distribution": doc_type_distribution,
            "success_by_document_type": doc_type_avg_success,
            "current_method_scores": self.method_scores,
            "dependencies_count": len(self.input_dependencies),
            "tools_count": len(self.tools_dataloaders),
            "total_decisions": len(self.decision_history),
            "processing_time": getattr(self, '_last_processing_time', 0.0),
            "method_used": getattr(self, '_last_method_used', None),
            "success": getattr(self, '_last_success', False)
        }

    def query_results(self, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query results from the database.
        
        Args:
            filename: Optional filename to filter results
            
        Returns:
            List of dictionaries containing extraction results
        """
        conn = sqlite3.connect('pdf_extraction.db')
        c = conn.cursor()
        
        if filename:
            c.execute('SELECT * FROM pdf_extractions WHERE filename = ?', (filename,))
        else:
            c.execute('SELECT * FROM pdf_extractions')
            
        results = []
        for row in c.fetchall():
            result = {
                'id': row[0],
                'filename': row[1],
                'document_type': row[2],
                'method_used': row[3],
                'rationale': row[4],
                'characteristics': json.loads(row[5]) if row[5] else {},
                'extracted_text': json.loads(row[6]) if row[6] else [],
                'metadata': json.loads(row[7]) if row[7] else {},
                'created_at': row[8]
            }
            results.append(result)
            
        conn.close()
        return results

    def _determine_document_type(self, characteristics: Dict[str, float]) -> str:
        """
        Determine document type based on characteristics.
        
        Args:
            characteristics: Dictionary with document characteristics
            
        Returns:
            Document type classification
        """
        # Extract key metrics
        text_density = characteristics["text_density"]
        image_density = characteristics["image_density"]
        table_indicators = characteristics["table_indicators"]
        scanned_indicators = characteristics["scanned_indicators"]
        
        # Get thresholds
        thresholds = self.characteristic_thresholds
        
        # Determine document type based on characteristics
        if scanned_indicators > thresholds["scanned_indicators"]["high"]:
            return "scanned"
        elif table_indicators > thresholds["table_indicators"]["high"]:
            return "table_heavy"
        elif image_density > thresholds["image_density"]["high"]:
            return "image_heavy"
        elif text_density > thresholds["text_density"]["high"]:
            return "digital_text"
        else:
            return "mixed"


# Base class for input dependencies
class InputDependency:
    """Base class for input dependencies that can be added to the PDF processing chain."""
    
    def process(self, input_path: str) -> str:
        """
        Process the input and return the path to the processed file.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Path to the processed file
        """
        raise NotImplementedError("Subclasses must implement process method")


# Example implementations of input dependencies

class ImagePreprocessor(InputDependency):
    """Dependency for preprocessing images in PDFs."""
    
    def __init__(self, contrast_boost=1.5, brightness_boost=1.2):
        self.contrast_boost = contrast_boost
        self.brightness_boost = brightness_boost
    
    def process(self, input_path: str) -> str:
        """Apply contrast and brightness enhancement to PDF pages."""
        # Implementation would extract images, enhance them, and create a new PDF
        # This is a placeholder that would actually modify the PDF
        print(f"ImagePreprocessor: Enhancing images in {input_path}")
        return input_path  # Return the path to the processed file


class OCRPreprocessor(InputDependency):
    """Dependency for preprocessing PDFs for OCR."""
    
    def __init__(self, dpi=300, binarize=True):
        self.dpi = dpi
        self.binarize = binarize
    
    def process(self, input_path: str) -> str:
        """Optimize PDF for OCR processing."""
        # Implementation would optimize the PDF for OCR
        # This is a placeholder that would actually modify the PDF
        print(f"OCRPreprocessor: Optimizing {input_path} for OCR at {self.dpi} DPI")
        return input_path  # Return the path to the processed file


# Example usage
def extract_text_from_pdf(pdf_path: str, extraction_priority: Optional[str] = None) -> List[str]:
    """
    Helper function to extract text from a PDF file using the decision agent.
    
    Args:
        pdf_path: Path to the PDF file
        extraction_priority: Priority for extraction ("speed", "accuracy", "tables", "images")
        
    Returns:
        List of extracted text by page
    """
    # Create agent with dependencies
    image_preprocessor = ImagePreprocessor(contrast_boost=1.8)
    ocr_preprocessor = OCRPreprocessor(dpi=300)
    
    agent = PDFProcessingDecisionAgent()
    
    # Add dependencies
    agent.add_input_dependency(image_preprocessor)
    agent.add_input_dependency(ocr_preprocessor)
    
    # Register custom tools
    agent.register_tool("extract_table_structure", lambda pdf: ["Table structure extraction would happen here"])
    
    # Choose method and process
    selection = agent.select_method(pdf_path, extraction_priority)
    print(f"Selected method: {selection['method']}")
    print(f"Rationale: {selection['rationale']}")
    
    documents = agent.process_pdf(pdf_path, extraction_priority, use_dependencies=True)
    
    # Extract text from documents
    extracted_text = [doc.text for doc in documents]
    
    return extracted_text 