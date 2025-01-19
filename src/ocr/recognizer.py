import cv2
import pytesseract
from typing import Tuple
import logging

class OCRRecognizer:
    """Handles OCR operations for car plate recognition."""
    
    def __init__(self, config: dict):
        self.config = config['ocr']['tesseract']
        self.logger = logging.getLogger(__name__)
        
    def recognize(self, image) -> Tuple[str, float]:
        """
        Perform OCR on preprocessed image.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Tuple containing recognized text and confidence score
        """
        try:
            custom_config = (
                f'--oem 3 --psm {self.config["psm"]} '
                f'-c tessedit_char_whitelist={self.config["whitelist"]}'
            )
            
            ocr_result = pytesseract.image_to_string(
                image,
                lang=self.config['lang'],
                config=custom_config
            )
            
            ocr_result_cleaned = ocr_result.strip()
            confidence = float(len(ocr_result_cleaned) > 0)
            
            self.logger.info(f"OCR Result: {ocr_result_cleaned}")
            return ocr_result_cleaned, confidence
            
        except Exception as e:
            self.logger.error(f"OCR failed: {str(e)}")
            raise OCRError(f"Failed to perform OCR: {str(e)}")

