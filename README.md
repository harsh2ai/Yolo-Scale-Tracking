## Video Setup Instructions

### 1. Download Sample Videos
Download the sample traffic videos from our Google Drive:
https://drive.google.com/drive/folders/1twBX1e4ALyZTvsqnw7tJSdgaW9Gpfj0d?usp=sharing

### 2. Video File Structure
After downloading, organize the videos as follows:

```
Yolo-Scale-Tracking/
├── videos/
│   ├── 4K Video of Highway Traffic!.mp4
│   ├── [appsgolem.com][00-00-00][00-01-40]_Road_traffic_video_for_object_recognition.mp4
│   └── other_traffic_videos.mp4
```

Steps:
1. Create a `videos` directory in your project root if it doesn't exist:
   ```bash
   mkdir videos
   ```

2. Move all downloaded video files into the `videos` directory
   - All video files should be directly inside the `videos` folder
   - Do not create subdirectories
   - Supported formats: .mp4, .avi, .mov

Your final project structure should look like:
```
Yolo-Scale-Tracking/
├── src/
│   ├── handlers/
│   ├── processors/
│   └── utils/
├── videos/              # Place downloaded videos here
├── output/             # Will store processed output
├── logs/               # Will store processing logs
├── main.py
├── setup.py
└── README.md
```

### 3. Verify Setup
- Ensure all video files are in the correct location
- Check video file permissions are correct
- Make sure no videos exceed GitHub's file size limit if you plan to commit them

### Note
- The `output` directory will be created automatically when you run the program
- Processed videos will be saved with '_processed' suffix in their names
- Original videos remain unchanged in the `videos` directory

### Troubleshooting
- If videos don't appear in the UI, check they are in the correct directory
- Ensure video files have correct read permissions
- Verify video formats are supported (.mp4, .avi, .mov)