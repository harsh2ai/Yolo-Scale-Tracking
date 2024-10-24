#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import time
from pathlib import Path
import zipfile
import shutil

class SetupManager:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.env_name = "yolo"
        self.python_version = "3.10"
        self.required_dirs = ['videos', 'output', 'logs', 'processed_videos']
        self.zip_name = "videos.zip"  # Name of the zip file containing videos
        self.colors = {
            'GREEN': '\033[92m',
            'BLUE': '\033[94m',
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'ENDC': '\033[0m'
        }

    def print_colored(self, text, color):
        """Print colored text."""
        print(f"{self.colors[color]}{text}{self.colors['ENDC']}")

    def run_command(self, command, shell=False):
        """Run a command and return its output."""
        try:
            if isinstance(command, str) and not shell:
                command = command.split()
            result = subprocess.run(command, 
                                 shell=shell, 
                                 check=True, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    def check_and_extract_videos(self):
        """Check for videos.zip and extract if found."""
        self.print_colored("\nChecking for video files...", "BLUE")
        
        zip_path = Path(self.zip_name)
        videos_dir = Path('videos')
        
        if zip_path.exists():
            self.print_colored("Found videos.zip, extracting...", "BLUE")
            try:
                # Create videos directory if it doesn't exist
                videos_dir.mkdir(exist_ok=True)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(videos_dir)
                
                self.print_colored("Videos extracted successfully!", "GREEN")
                
                # Move any videos from subdirectories to main videos directory
                for root, dirs, files in os.walk(videos_dir):
                    for file in files:
                        if file.endswith(('.mp4', '.avi', '.mov')):
                            src_path = Path(root) / file
                            dst_path = videos_dir / file
                            if src_path != dst_path:
                                shutil.move(str(src_path), str(dst_path))
                
                # Clean up empty subdirectories
                for root, dirs, files in os.walk(videos_dir, topdown=False):
                    for dir_name in dirs:
                        dir_path = Path(root) / dir_name
                        try:
                            dir_path.rmdir()  # Will only remove if empty
                        except OSError:
                            pass
                
            except Exception as e:
                self.print_colored(f"Error extracting videos: {str(e)}", "RED")
                return False
        else:
            self.print_colored("No videos.zip found. Please add videos manually to the 'videos' directory.", "YELLOW")
        
        return True

    def setup_directory_structure(self):
        """Create necessary directories."""
        self.print_colored("\nSetting up directory structure...", "BLUE")
        
        for directory in self.required_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                self.print_colored(f"Created directory: {directory}", "GREEN")
            else:
                self.print_colored(f"Directory already exists: {directory}", "YELLOW")

    # ... [Previous methods remain the same] ...

    def verify_setup(self):
        """Verify the setup is complete and correct."""
        self.print_colored("\nVerifying setup...", "BLUE")
        
        # Check directories
        all_good = True
        for directory in self.required_dirs:
            if not Path(directory).exists():
                self.print_colored(f"Missing directory: {directory}", "RED")
                all_good = False
        
        # Check for videos
        videos_dir = Path('videos')
        video_files = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi')) + list(videos_dir.glob('*.mov'))
        if not video_files:
            self.print_colored("No video files found in videos directory!", "YELLOW")
        else:
            self.print_colored(f"Found {len(video_files)} video files", "GREEN")
            for video in video_files:
                print(f"  - {video.name}")
        
        return all_good

    def cleanup_temporary_files(self):
        """Clean up any temporary files created during setup."""
        try:
            # Add any temporary files or directories to clean up here
            temp_files = []
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            self.print_colored(f"Error during cleanup: {str(e)}", "YELLOW")

    def display_completion_message(self):
        """Display completion message with instructions."""
        self.print_colored("\n" + "="*50, "GREEN")
        self.print_colored("Setup completed successfully!", "GREEN")
        self.print_colored("="*50 + "\n", "GREEN")
        
        # Count videos
        video_count = len(list(Path('videos').glob('*.mp4')))
        self.print_colored(f"Found {video_count} videos ready for processing\n", "BLUE")
        
        self.print_colored("To start using the system:", "BLUE")
        print("\n1. Activate the conda environment:")
        if self.os_type == "windows":
            print("   conda activate yolo")
        else:
            print("   source activate yolo")
        
        print("\n2. Run the main script:")
        print("   python main.py")
        
        print("\nDirectory structure created:")
        for directory in self.required_dirs:
            print(f"   - {directory}/")
        
        print("\nOptional arguments:")
        print("   --model yolov8m.pt")
        print("   --batch-size 16")
        print("   --conf-threshold 0.35")
        print("   --save-output")
        
        self.print_colored("\nEnjoy using the Multi-Video Object Detection System!", "GREEN")

def main():
    setup = SetupManager()
    
    # Print header
    setup.print_colored("\n=== Multi-Video Object Detection System Setup ===\n", "BLUE")
    
    try:
        # Setup directory structure
        setup.setup_directory_structure()
        
        # Extract videos if zip exists
        setup.check_and_extract_videos()
        
        # Verify setup
        if setup.verify_setup():
            setup.print_colored("Directory structure verified!", "GREEN")
        
        # Continue with conda environment setup and package installation
        setup.check_conda()
        if setup.create_conda_env():
            setup.activate_and_install()
            setup.verify_cuda()
        
        # Final verification and cleanup
        setup.cleanup_temporary_files()
        setup.display_completion_message()
        
    except KeyboardInterrupt:
        setup.print_colored("\nSetup interrupted by user.", "YELLOW")
        sys.exit(1)
    except Exception as e:
        setup.print_colored(f"\nError during setup: {str(e)}", "RED")
        sys.exit(1)

if __name__ == "__main__":
    main()