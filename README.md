# Multi-Video Object Detection System

A scalable system for processing multiple video streams simultaneously using YOLOv8 object detection. The system automatically closes video windows upon completion and supports GPU acceleration.

## Features

- Multi-video processing with YOLOv8
- Automatic window management (closes when video completes)
- GPU acceleration support
- Batch processing capabilities
- Progress tracking and logging
- Configurable detection parameters
- Support for multiple video formats

## Directory Structure

```
project_root/
├── src/
│   ├── handlers/
│   │   └── video_stream_handler.py
│   ├── processors/
│   │   └── video_processor.py
│   └── utils/
│       └── logger.py
├── videos/             # Place input videos here
├── output/            # Processed videos will be saved here
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone 
cd 
```

2. Create and activate a virtual environment (recommended):
```bash
# Using conda
conda create -n yolo python=3.10
conda activate yolo

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your videos in the `videos` directory
2. Run with default settings:
```bash
python main.py
```

### Advanced Usage

Customize processing with various command-line arguments:

```bash
python main.py \
    --video-dir videos \
    --model yolov8m.pt \
    --batch-size 16 \
    --conf-threshold 0.35 \
    --save-output \
    --output-dir processed_videos \
    --buffer-size 50
```

### Command-line Arguments

|
 Argument 
|
 Description 
|
 Default 
|
|
----------
|
-------------
|
---------
|
|
 --video-dir 
|
 Directory containing video files 
|
 videos 
|
|
 --model 
|
 YOLOv8 model to use 
|
 yolov8n.pt 
|
|
 --batch-size 
|
 Batch size for processing 
|
 8 
|
|
 --conf-threshold 
|
 Confidence threshold for detections 
|
 0.25 
|
|
 --save-output 
|
 Save processed videos 
|
 False 
|
|
 --buffer-size 
|
 Frame buffer size per video 
|
 30 
|
|
 --output-dir 
|
 Directory to save processed videos 
|
 output 
|

## Models Available

- yolov8n.pt (fastest)
- yolov8s.pt
- yolov8m.pt
- yolov8l.pt
- yolov8x.pt (most accurate)

## Performance Tips

1. **GPU Memory Usage**:
   - Adjust batch size based on your GPU memory
   - Larger batch sizes generally mean better performance
   - Start with batch_size=8 and adjust as needed

2. **Processing Speed**:
   - Use smaller models (yolov8n.pt) for faster processing
   - Reduce video resolution if needed
   - Increase buffer size for smoother processing

3. **Detection Quality**:
   - Use larger models (yolov8l.pt, yolov8x.pt) for better accuracy
   - Adjust confidence threshold based on your needs
   - Higher confidence threshold means fewer but more accurate detections

## Supported Video Formats

- .mp4
- .avi
- .mov

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB RAM minimum (16GB recommended)
- NVIDIA drivers and CUDA toolkit for GPU support

## Troubleshooting

1. **GPU Not Detected**:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `torch.cuda.is_available()`
   - Update GPU drivers

2. **Out of Memory**:
   - Reduce batch size
   - Use a smaller model
   - Process fewer videos simultaneously

3. **Poor Performance**:
   - Increase batch size if GPU memory allows
   - Check CPU/GPU utilization
   - Adjust buffer size

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team