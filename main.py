# main.py
import sys
import os
import torch
import argparse
from pathlib import Path
from src.utils.logger import setup_logger
from src.processors.video_processor import VideoProcessor

def get_video_files(video_dir: str, extensions=('.mp4', '.avi', '.mov')):
    """Get all video files from directory."""
    video_dir = Path(video_dir)
    video_files = []
    for ext in extensions:
        video_files.extend(list(video_dir.glob(f'*{ext}')))
    return [str(f) for f in video_files]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-Video Object Detection Processing')
    
    # Required arguments
    parser.add_argument('--video-dir', type=str, default='videos',
                      help='Directory containing video files (default: videos)')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                      help='YOLOv8 model to use (default: yolov8n.pt)')
    
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for processing (default: 8)')
    
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                      help='Confidence threshold for detections (default: 0.25)')
    
    parser.add_argument('--save-output', action='store_true',
                      help='Save processed videos (default: True)')
    
    parser.add_argument('--buffer-size', type=int, default=30,
                      help='Frame buffer size per video (default: 30)')
    
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save processed videos (default: output)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger()
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s):")
            for i in range(torch.cuda.device_count()):
                print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Using CPU.")

        # Create output directory if saving videos
        if args.save_output:
            os.makedirs(args.output_dir, exist_ok=True)

        # Get video files from directory
        video_paths = get_video_files(args.video_dir)
        if not video_paths:
            logger.error(f"No video files found in directory: {args.video_dir}")
            sys.exit(1)

        logger.info(f"Found {len(video_paths)} video files:")
        for path in video_paths:
            logger.info(f"  - {path}")
        
        # Create and run video processor
        processor = VideoProcessor(
            video_paths=video_paths,
            model_path=args.model,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            save_output=args.save_output,
            buffer_size=args.buffer_size
        )
        
        processor.run()
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()