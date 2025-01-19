# src/exceptions.py

class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    def __init__(self, message: str = "PDF processing failed"):
        self.message = message
        super().__init__(self.message)

class OCRError(Exception):
    """Raised when OCR processing fails."""
    def __init__(self, message: str = "OCR processing failed"):
        self.message = message
        super().__init__(self.message)

class ImageProcessingError(Exception):
    """Raised when image processing fails."""
    def __init__(self, message: str = "Image processing failed"):
        self.message = message
        super().__init__(self.message)

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    def __init__(self, message: str = "Configuration error"):
        self.message = message
        super().__init__(self.message)

class EmailError(Exception):
    """Raised when email operations fail."""
    def __init__(self, message: str = "Email operation failed"):
        self.message = message
        super().__init__(self.message)

class DownloadError(Exception):
    """Raised when file download fails."""
    def __init__(self, message: str = "Download failed"):
        self.message = message
        super().__init__(self.message)

