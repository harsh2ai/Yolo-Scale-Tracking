# src/processors/video_processor.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from typing import List, Dict
import threading
from queue import Queue, Empty
import logging
from ..handlers.video_stream_handler import VideoStreamHandler
from ..utils.logger import VideoStreamError

class VideoProcessor:
    def __init__(self, video_paths: List[str], model_path: str, 
                 batch_size: int = 8,
                 conf_threshold: float = 0.25, 
                 save_output: bool = True,
                 buffer_size: int = 30):
        
        self.video_paths = video_paths
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.save_output = save_output
        self.buffer_size = buffer_size
        
        # Track active videos
        self.active_videos = {i: True for i in range(len(video_paths))}
        self.completed_videos = set()
        
        # Initialize components
        self.device = self.setup_gpu()
        self.model = self.initialize_model(model_path)
        self.streams = self.setup_streams()
        self.writers = self.setup_writers()
        
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self.processing_errors = {i: 0 for i in range(len(video_paths))}
        self.max_errors = 5

    def setup_gpu(self):
        """Setup GPU and CUDA optimizations."""
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                device = torch.device('cuda:0')
                logging.info(f"GPU setup complete. Using: {torch.cuda.get_device_name(0)}")
                return device
            else:
                logging.warning("CUDA not available, using CPU")
                return torch.device('cpu')
        except Exception as e:
            logging.error(f"Error setting up GPU: {str(e)}")
            return torch.device('cpu')

    def initialize_model(self, model_path: str):
        """Initialize YOLO model."""
        try:
            logging.info("Loading YOLO model...")
            model = YOLO(model_path)
            if torch.cuda.is_available():
                model.to(self.device)
            logging.info(f"YOLO model loaded successfully on {self.device}")
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            raise

    def setup_streams(self):
        """Initialize video streams."""
        streams = []
        for path in self.video_paths:
            try:
                stream = VideoStreamHandler(path, self.buffer_size)
                streams.append(stream)
                logging.info(f"Initialized stream for {path}")
            except Exception as e:
                logging.error(f"Error setting up video stream for {path}: {str(e)}")
                # Cleanup already created streams
                for s in streams:
                    s.release()
                raise
        return streams

    def setup_writers(self):
        """Initialize video writers."""
        writers = []
        if self.save_output:
            for i, path in enumerate(self.video_paths):
                try:
                    output_path = path.rsplit('.', 1)[0] + '_processed.mp4'
                    stream = self.streams[i]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    width = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(stream.cap.get(cv2.CAP_PROP_FPS))
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    writers.append(writer)
                    logging.info(f"Initialized writer for {output_path}")
                except Exception as e:
                    logging.error(f"Error creating video writer for {path}: {str(e)}")
                    # Cleanup
                    for w in writers:
                        w.release()
                    raise
        return writers

    def process_frames(self):
        """Process frames with error handling."""
        while True:
            try:
                if self.input_queue.empty():
                    if all(not stream.running() for stream in self.streams):
                        break
                    time.sleep(0.001)
                    continue
                
                # Collect and process batch
                batch_frames = []
                batch_info = []
                
                while len(batch_frames) < self.batch_size:
                    try:
                        frame_data = self.input_queue.get(timeout=0.1)
                        if frame_data['frame'] is not None:
                            batch_frames.append(frame_data['frame'])
                            batch_info.append(frame_data)
                    except Empty:
                        break
                
                if not batch_frames:
                    continue
                
                # Process batch with error handling
                try:
                    with torch.inference_mode():
                        results = self.model(batch_frames, conf=self.conf_threshold)
                    
                    # Process results
                    for i, (result, info) in enumerate(zip(results, batch_info)):
                        try:
                            processed_frame = self.draw_detections(batch_frames[i], result)
                            self.output_queue.put({
                                'frame': processed_frame,
                                'stream_id': info['stream_id'],
                                'frame_count': info['frame_count']
                            })
                            self.processing_errors[info['stream_id']] = 0
                        except Exception as e:
                            logging.error(f"Error processing frame {info['frame_count']} "
                                        f"for stream {info['stream_id']}: {str(e)}")
                            self.processing_errors[info['stream_id']] += 1
                            
                except Exception as e:
                    logging.error(f"Error in batch processing: {str(e)}")
                    for info in batch_info:
                        self.processing_errors[info['stream_id']] += 1
                
            except Exception as e:
                logging.error(f"Error in process_frames: {str(e)}")
                time.sleep(0.1)

    def draw_detections(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw detection boxes with error handling."""
        try:
            processed_frame = frame.copy()
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    class_name = self.model.names[int(cls)]
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f'{class_name} {conf:.2f}'
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(processed_frame, (x1, y1-label_height-10),
                                (x1+label_width, y1), (0, 255, 0), -1)
                    cv2.putText(processed_frame, label, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            return processed_frame
        except Exception as e:
            logging.error(f"Error drawing detections: {str(e)}")
            return frame

    def check_video_completion(self, stream_id: int, frame_count: int):
        """Check if a video has completed processing and close its window"""
        try:
            if stream_id in self.active_videos and self.active_videos[stream_id]:
                total_frames = self.streams[stream_id].total_frames
                if frame_count >= total_frames - 1:
                    logging.info(f"Video {stream_id + 1} completed processing")
                    
                    # Close specific window
                    cv2.destroyWindow(f"Video {stream_id + 1}")
                    
                    # Update status
                    self.active_videos[stream_id] = False
                    self.completed_videos.add(stream_id)
                    
                    # Release writer if exists
                    if self.save_output and stream_id < len(self.writers):
                        if self.writers[stream_id] is not None:
                            self.writers[stream_id].release()
                            self.writers[stream_id] = None
                    
                    logging.info(f"Closed window and released resources for Video {stream_id + 1}")
                    
        except Exception as e:
            logging.error(f"Error in check_video_completion for stream {stream_id}: {str(e)}")

    def display_and_save_frame(self, output: Dict, windows: List[str], start_time: float):
        """Display and save processed frame with auto-closing."""
        try:
            stream_id = output['stream_id']
            frame = output['frame']
            frame_count = output['frame_count']
            
            if stream_id in self.completed_videos:
                return
            
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                progress = (frame_count / self.streams[stream_id].total_frames) * 100
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                logging.info(f"Video {stream_id+1} Progress: {progress:.1f}% "
                          f"({frame_count}/{self.streams[stream_id].total_frames}), "
                          f"FPS: {fps:.1f}, GPU Memory: {gpu_mem:.2f}GB")
            
            if self.active_videos[stream_id]:
                window_name = windows[stream_id]
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, frame)
                
                if self.save_output and stream_id < len(self.writers) and self.writers[stream_id] is not None:
                    try:
                        self.writers[stream_id].write(frame)
                    except Exception as e:
                        logging.error(f"Error saving frame for stream {stream_id}: {str(e)}")
            
            self.check_video_completion(stream_id, frame_count)
                    
        except Exception as e:
            logging.error(f"Error in display_and_save_frame: {str(e)}")

    def run(self):
        """Main processing loop with auto-closing."""
        start_time = time.time()
        windows = [f"Video {i+1}" for i in range(len(self.video_paths))]
        
        try:
            for stream in self.streams:
                stream.start()
            
            process_thread = threading.Thread(target=self.process_frames)
            process_thread.start()
            
            while True:
                try:
                    if len(self.completed_videos) == len(self.video_paths):
                        logging.info("All videos completed processing")
                        break
                    
                    for i, stream in enumerate(self.streams):
                        if self.active_videos[i] and stream.running():
                            frame = stream.read()
                            
                            if frame is not None:
                                self.input_queue.put({
                                    'frame': frame,
                                    'stream_id': i,
                                    'frame_count': stream.frame_count
                                })
                    
                    while not self.output_queue.empty():
                        output = self.output_queue.get()
                        self.display_and_save_frame(output, windows, start_time)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    time.sleep(0.001)
                        
                except Exception as e:
                    logging.error(f"Error in main loop: {str(e)}")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
            
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        try:
            for stream in self.streams:
                stream.release()
            
            for writer in self.writers:
                if writer is not None:
                    writer.release()
            
            cv2.destroyAllWindows()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logging.info("Cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")