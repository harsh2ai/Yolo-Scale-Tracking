import logging
from datetime import datetime
import sys

def setup_logger():
    """Configure and setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'video_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class VideoStreamError(Exception):
    """Custom exception for video stream errors"""
    pass