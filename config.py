# config.py - Updated with working PostgreSQL configuration
import os

DATABASE_CONFIG = {
    'url': 'postgresql://omkarsarkar@localhost:5432/vcare_wellness',  # Updated with your username
    'enable_timescale': False,  # Disable TimescaleDB for now
    'pool_size': 20,
    'max_overflow': 30
}

# Video configuration with fallback options
VIDEO_CONFIG = {
    # Try these paths in order
    'video_paths': [
        "/Users/omkarsarkar/Desktop/Vyom/mavni-1/takeoff.mp4",  # Original path
        "./takeoff.mp4",  # Current directory
        "./videos/takeoff.mp4",  # Videos folder
        "./sample_video.mp4",  # Alternative name
        0  # Webcam fallback (if available)
    ],
    'default_webcam': 0,  # Default webcam device
    'frame_skip': 2,  # Process every 2nd frame for performance
    'video_quality': 70,  # JPEG encoding quality (0-100)
}

# System paths
PATHS_CONFIG = {
    'reports_dir': "reports",
    'static_files_dir': "static",
    'models_dir': "models",
    'temp_dir': "temp"
}

# Monitoring settings
MONITORING_CONFIG = {
    'data_retention_hours': 72,
    'alert_cooldown_seconds': 300,
    'db_save_interval_seconds': 5,
    'chart_history_seconds': 60
}