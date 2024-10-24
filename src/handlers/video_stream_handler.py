import cv2
import numpy as np
import threading
from queue import Queue, Empty, Full
import time
import logging
from typing import Optional, Dict
from ..utils.logger import VideoStreamError

class VideoStreamHandler:
    def __init__(self, video_path: str, buffer_size: int = 30, retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Initialize Video Stream Handler with error handling and retry mechanism.
        """
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Initialize internal state
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.total_frames = 0
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False
        self.error_count = 0
        self.last_frame = None
        self.last_successful_read = time.time()
        self.width = 0
        self.height = 0
        
        # Initialize video capture with retry mechanism
        self._initialize_capture()

    def _initialize_capture(self) -> bool:
        """Initialize video capture with retry mechanism."""
        for attempt in range(self.retry_attempts):
            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise VideoStreamError(f"Failed to open video: {self.video_path}")
                
                # Get video properties
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Verify valid properties
                if self.fps <= 0 or self.width <= 0 or self.height <= 0:
                    raise VideoStreamError(f"Invalid video properties: FPS={self.fps}, "
                                        f"Width={self.width}, Height={self.height}")
                
                # Set buffer size for video capture
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                logging.info(f"Successfully initialized video capture for {self.video_path}")
                logging.info(f"Video properties: {self.width}x{self.height} @ {self.fps}fps")
                return True
                
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{self.retry_attempts} failed: {str(e)}")
                if self.cap:
                    self.cap.release()
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise VideoStreamError(f"Failed to initialize video capture after {self.retry_attempts} attempts")
        
        return False

    def _check_frame_validity(self, frame: np.ndarray) -> bool:
        """Check if frame is valid."""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        if np.any(np.isnan(frame)):
            return False
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            return False
        return True

    def _handle_frame_error(self) -> Optional[np.ndarray]:
        """Handle frame reading errors."""
        self.error_count += 1
        logging.warning(f"Frame read error #{self.error_count} for {self.video_path}")
        
        if time.time() - self.last_successful_read > 5.0:
            logging.info("Attempting to reconnect video stream...")
            self.cap.release()
            if self._initialize_capture():
                self.error_count = 0
                self.last_successful_read = time.time()
                
        if self.last_frame is not None:
            return self.last_frame.copy()
        return None

    def start(self) -> 'VideoStreamHandler':
        """Start the video stream thread."""
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Update the frame buffer continuously."""
        while not self.stopped:
            if self.buffer.full():
                time.sleep(0.001)
                continue
                
            try:
                ret, frame = self.cap.read()
                
                if not ret or not self._check_frame_validity(frame):
                    frame = self._handle_frame_error()
                    if frame is None:
                        if self.error_count > self.retry_attempts:
                            logging.error(f"Too many consecutive errors for {self.video_path}")
                            self.stopped = True
                        continue
                else:
                    self.error_count = 0
                    self.last_successful_read = time.time()
                    self.last_frame = frame.copy()
                
                try:
                    self.buffer.put(frame, timeout=1)
                    self.frame_count += 1
                    
                    if self.frame_count % 100 == 0:
                        progress = (self.frame_count / self.total_frames) * 100
                        logging.debug(f"Stream progress: {progress:.1f}% "
                                   f"({self.frame_count}/{self.total_frames})")
                        
                except Full:
                    logging.warning("Buffer full, skipping frame")
                    continue
                    
            except Exception as e:
                logging.error(f"Error in video stream update: {str(e)}")
                if self.error_count > self.retry_attempts:
                    self.stopped = True
                    break
                time.sleep(self.retry_delay)

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the buffer."""
        try:
            return self.buffer.get(timeout=1)
        except Empty:
            return None

    def get_stream_info(self) -> Dict:
        """Get information about the video stream."""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'current_frame': self.frame_count,
            'resolution': (self.width, self.height),
            'is_running': self.running(),
            'error_count': self.error_count,
            'buffer_size': self.buffer.qsize()
        }

    def running(self) -> bool:
        """Check if the stream is still running."""
        return not self.stopped or not self.buffer.empty()

    def release(self):
        """Release all resources."""
        self.stopped = True
        if self.cap:
            self.cap.release()
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
        logging.info(f"Released video stream: {self.video_path}")