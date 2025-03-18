import os
import logging
from pdf_agent.agent import PDFProcessingDecisionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the PDF processing agent
    agent = PDFProcessingDecisionAgent(
        name="pdf_processor",
        skills=["pdf_extraction", "ocr", "document_analysis"],
        goal="Extract and analyze text from PDF documents efficiently",
        sampling_pages=2
    )
    
    # Directory containing PDF files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(current_dir, "test_pdfs")
    
    # Process each PDF in the directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            logger.info(f"\nProcessing {filename}...")
            
            try:
                # Process the PDF
                documents = agent.process_pdf(pdf_path)
                
                # Log results
                logger.info(f"Successfully processed {filename}")
                logger.info(f"Extracted text from {len(documents)} pages")
                
                # Query results from database
                results = agent.query_results(filename=filename)
                logger.info(f"Found {len(results)} results in database")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    
    # Print summary statistics
    logger.info("\n=== Processing Summary ===")
    
    # Get all results
    all_results = agent.query_results()
    logger.info(f"Total documents processed: {len(all_results)}")
    
    # Count by document type
    doc_types = {}
    for result in all_results:
        doc_type = result["document_type"]
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    logger.info("\nDocument Type Distribution:")
    for doc_type, count in doc_types.items():
        logger.info(f"{doc_type}: {count}")
    
    # Count by method used
    methods = {}
    for result in all_results:
        method = result["method_used"]
        methods[method] = methods.get(method, 0) + 1
    
    logger.info("\nMethod Usage Distribution:")
    for method, count in methods.items():
        logger.info(f"{method}: {count}")

if __name__ == "__main__":
    main() 