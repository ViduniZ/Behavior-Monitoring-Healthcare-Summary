import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()


class PatientActivityMonitor:
    def __init__(self, model_name='yolo11n.pt', confidence=0.6, db_config=None):
        """Initialize the Patient Activity Monitor"""
        print("🏥 Initializing Patient Activity Monitoring System...")
        
        # Load YOLO11 model
        print(f"📦 Loading {model_name}...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        
        # Activity detection thresholds
        self.activity_classes = {
            'bottle': 'drinking',
            'cup': 'drinking',
            'wine glass': 'drinking',
            'fork': 'eating',
            'knife': 'eating',
            'spoon': 'eating',
            'bowl': 'eating',
            'banana': 'eating',
            'apple': 'eating',
            'sandwich': 'eating',
            'orange': 'eating',
            'pizza': 'eating',
            'cake': 'eating',
            'person': 'face_presence'
        }
        
        # Motion detection parameters
        self.previous_frame = None
        self.motion_threshold = 20
        self.min_motion_area = 800
        
        # Activity tracking
        self.current_activities = set()
        self.activity_history = deque(maxlen=100)
        self.last_activity_time = {}
        self.activity_durations = defaultdict(int)
        
        # Face detection tracking (prevent counting every frame)
        self.face_currently_detected = False
        self.last_face_detection_time = None
        self.face_detection_cooldown = 5  # seconds between face detection logs
        
        # Patient condition monitoring
        self.motion_history = deque(maxlen=60)  # Last 60 frames (~3 seconds)
        self.face_detection_history = deque(maxlen=60)
        self.condition_status = "Unknown"
        self.no_motion_duration = 0
        self.alert_triggered = False
        self.alert_cooldown = 30  # seconds
        self.last_alert_time = 0
        