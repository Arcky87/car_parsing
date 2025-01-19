import yaml
import logging
from pathlib import Path
import gdown
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .processors import PDFProcessor, ImageProcessor, DocumentGenerator, EmailSender
from .exceptions import (
    PDFProcessingError, 
    OCRError, 
    ImageProcessingError, 
    ConfigurationError,
    EmailError,
    DownloadError
)

def load_config() -> Dict:
    """
    Load configuration from config.yaml file.
    
    Returns:
        Dict: Configuration dictionary
    
    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    try:
        config_path = Path('config/config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")

def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def download_files(config: Dict) -> Path:
    """
    Download files from Google Drive.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path: Path to download directory
    
    Raises:
        DownloadError: If download fails
    """
    try:
        download_dir = Path(config['google_drive']['download_dir'])
        download_dir.mkdir(parents=True, exist_ok=True)
        
        gdown.download_folder(
            config['google_drive']['folder_url'],
            output=str(download_dir),
            use_cookies=False,
            resume=True
        )
        
        logging.info(f"Files downloaded to {download_dir}")
        return download_dir
    except Exception as e:
        raise DownloadError(f"Failed to download files: {str(e)}")

def process_violations(pdf_files: List[Path], processors: Dict) -> List[Dict]:
    """
    Process violation PDFs and collect results.
    
    Args:
        pdf_files: List of PDF file paths
        processors: Dictionary of processor instances
    
    Returns:
        List[Dict]: List of processed violation data
    """
    results = []
    for pdf_file in pdf_files:
        try:
            logging.info(f"Processing {pdf_file}")
            
            # Extract text and images
            text = processors['pdf'].extract_text(pdf_file)
            reg_photo, car_photo = processors['pdf'].extract_images(pdf_file)
            
            # Process registration number image
            processed_image = processors['image'].preprocess(reg_photo)
            
            # Get text information and OCR result
            text_info = processors['pdf'].get_text_info(text)
            ocr_result = processors['pdf'].tesseract_recognizer(processed_image)
            
            # Combine results
            result = {
                'filename': pdf_file.stem,
                'car_photo_filename': car_photo,
                'regnumber_filename': reg_photo,
                'ocr_result': ocr_result,
                **text_info
            }
            
            results.append(result)
            
            # Send alerts if needed
     #       if text_info['ИГР'] == ocr_result:
     #           if int(text_info['сумма_штрафа']) > config['email']['alerts']['fine_threshold']:
     #               processors['email'].send_violation_alert(result, 'cargo')
     #       else:
     #           processors['email'].send_violation_alert(result, 'tech')
                
        except (PDFProcessingError, ImageProcessingError, OCRError) as e:
            logging.error(f"Error processing {pdf_file}: {str(e)}")
            continue
            
    return results

def generate_reports(results: List[Dict], df: pd.DataFrame, doc_generator: DocumentGenerator) -> None:
    """Generate all required reports."""
    try:
        # Create violation table document
        doc_generator.create_table_document(results)
        
        # Create daily statistics document
        doc_generator.create_daily_statistics(df)
        
        logging.info("All reports generated successfully")
    except Exception as e:
        logging.error(f"Error generating reports: {str(e)}")
        raise

def main():
    """Main application function."""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting violation processing application")
        
        # Load configuration
        config = load_config()
        logging.info("Configuration loaded successfully")
        
        # Initialize processors
        processors = {
            'pdf': PDFProcessor(config),
            'image': ImageProcessor(config),
            'doc': DocumentGenerator(config),
            'email': EmailSender(config)
        }
        
        # Download files
        download_dir = download_files(config)
        
        # Get PDF files
        pdf_files = list(download_dir.rglob('*.pdf'))
        if not pdf_files:
            logging.warning("No PDF files found to process")
            return
        
        # Process violations
        results = process_violations(pdf_files, processors)
        if not results:
            logging.warning("No results generated from processing")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate reports
        generate_reports(results, df, processors['doc'])
        
        # Send daily statistics email
    #    if config['email']['alerts']['send_daily_stats']:
    #        processors['email'].send_daily_statistics(df)
        
        logging.info("Application completed successfully")
        
    except ConfigurationError as e:
        logging.error(f"Configuration error: {str(e)}")
    except DownloadError as e:
        logging.error(f"Download error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Application finished")

if __name__ == "__main__":
    main()

