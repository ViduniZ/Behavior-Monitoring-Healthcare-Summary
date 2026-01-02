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

# Statistics
        self.fps_counter = deque(maxlen=30)
        self.detection_confidence = {}
        self.session_stats = {
            'drinking_count': 0,
            'eating_count': 0,
            'motion_events': 0,
            'face_detections': 0,
            'alerts_triggered': 0
        }
        
        # Database connection
        self.db_config = db_config
        self.db_conn = None
        if db_config:
            self.connect_database()
        
        # Recording
        self.is_recording = False
        self.video_writer = None      
        
        print("✅ Patient Monitor initialized successfully!")
        print(f"   Model: {model_name}")
        print(f"   Confidence threshold: {confidence}")
        print(f"   Database: {'Connected' if self.db_conn else 'Disabled'}")
    
    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            print(f"🔄 Connecting to database at {self.db_config['host']}:{self.db_config['port']}...")
            self.db_conn = psycopg2.connect(**self.db_config)
            print("✅ Database connected successfully!")
            self.create_tables()
            return True
        except psycopg2.OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check if PostgreSQL is running")
            print("   2. Verify database credentials in .env file")
            print("   3. Ensure database exists")
            print("   4. Check firewall settings")
            self.db_conn = None
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.db_conn = None
            return False
    
    def create_tables(self):
        """Create necessary database tables"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            
            # Activity logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    activity_type VARCHAR(50) NOT NULL,
                    condition_status VARCHAR(20),
                    confidence_score FLOAT,
                    motion_intensity FLOAT,
                    details JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20),
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Session statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_stats (
                    id SERIAL PRIMARY KEY,
                    session_start TIMESTAMP NOT NULL,
                    session_end TIMESTAMP,
                    total_activities INT,
                    drinking_count INT,
                    eating_count INT,
                    motion_events INT,
                    alerts_count INT,
                    average_condition VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.db_conn.commit()
            cursor.close()
            print("✅ Database tables created/verified")
        except Exception as e:
            print(f"⚠️  Error creating tables: {e}")
    
    def log_activity(self, activity_type, condition_status, confidence, motion_intensity, details=None):
        """Log activity to database with NumPy-safe type conversion"""
        if not self.db_conn:
            return False

        # Convert NumPy numeric types to native Python types
        def to_native(value):
            if isinstance(value, (np.floating, np.float32, np.float64)):
                return float(value)
            if isinstance(value, (np.integer, np.int32, np.int64)):
                return int(value)
            return value

        try:
            confidence = to_native(confidence)
            motion_intensity = to_native(motion_intensity)

            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO activity_logs 
                (timestamp, activity_type, condition_status, confidence_score, motion_intensity, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                activity_type,
                condition_status,
                confidence,
                motion_intensity,
                json.dumps(details) if details else None
            ))

            self.db_conn.commit()
            cursor.close()
            return True

        except psycopg2.Error as e:
            print(f"⚠️  Database error logging activity: {e}")
            try:
                self.db_conn.rollback()
            except:
                pass
            return False

        except Exception as e:
            print(f"⚠️  Unexpected error logging activity: {e}")
            return False

    
    def log_alert(self, alert_type, severity, message):
        """Log alert to database"""
        if not self.db_conn:
            return False
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (%s, %s, %s, %s)
            """, (datetime.now(), alert_type, severity, message))
            self.db_conn.commit()
            cursor.close()
            return True
        except psycopg2.Error as e:
            print(f"⚠️  Database error logging alert: {e}")
            try:
                self.db_conn.rollback()
            except:
                pass
            return False
        except Exception as e:
            print(f"⚠️  Unexpected error logging alert: {e}")
            return False
        
    def detect_motion(self, frame):
        """Advanced motion detection with intensity calculation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return None, 0, [], False
        
        # Frame difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find motion contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:
                motion_areas.append(contour)
                total_motion_area += area
        
        # Calculate motion intensity (0-100 scale)
        frame_area = frame.shape[0] * frame.shape[1]
        motion_intensity = min(100, (total_motion_area / frame_area) * 1000)
        
        # Significant motion detected
        has_significant_motion = motion_intensity > 5
        
        self.previous_frame = gray
        
        return thresh, motion_intensity, motion_areas, has_significant_motion
    
    def detect_activities(self, frame):
        """Detect patient activities using YOLO"""
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detected_activities = set()
        face_detected = False
        max_confidences = {}
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Map to activity
            if class_name in self.activity_classes:
                activity = self.activity_classes[class_name]
                detected_activities.add(activity)
                
                # Track maximum confidence for each activity
                if activity not in max_confidences or confidence > max_confidences[activity]:
                    max_confidences[activity] = confidence
                
                if activity == 'face_presence':
                    face_detected = True
        
        self.detection_confidence = max_confidences
        
        return results, detected_activities, face_detected
    
    def assess_patient_condition(self, motion_intensity, face_detected, has_significant_motion):
        """Assess patient condition based on motion and face detection"""
        current_time = time.time()
        
        # Update history
        self.motion_history.append(motion_intensity)
        self.face_detection_history.append(face_detected)
        
        # Calculate average motion over last 3 seconds
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0
        face_presence_rate = np.mean(self.face_detection_history) if self.face_detection_history else 0
        
        # Condition assessment logic
        if face_presence_rate > 0.7:  # Face visible most of the time
            if avg_motion > 10:
                condition = "Good"
                self.no_motion_duration = 0
            elif avg_motion > 3:
                condition = "Fair"
                self.no_motion_duration = 0
            else:
                condition = "Poor"
                self.no_motion_duration += 1
        else:
            condition = "Unknown"
            self.no_motion_duration += 1
        
        # Generate alert for prolonged inactivity
        if self.no_motion_duration > 180 and not self.alert_triggered:  # ~9 seconds
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.trigger_alert("Low Activity", "High", 
                                 "Patient showing minimal movement for extended period")
                self.last_alert_time = current_time
        
        if has_significant_motion:
            self.alert_triggered = False
        
        self.condition_status = condition
        return condition
    
    def trigger_alert(self, alert_type, severity, message):
        """Trigger an alert"""
        self.alert_triggered = True
        self.session_stats['alerts_triggered'] += 1
        print(f"\n🚨 ALERT: {message}")
        self.log_alert(alert_type, severity, message)
    
    def process_activities(self, detected_activities):
        """Process and log detected activities with enhanced visible colors"""
        current_time = datetime.now()
        
        # ANSI color codes for terminal output
        COLORS = {
            'CYAN': '\033[96m',      # Bright cyan
            'GREEN': '\033[92m',     # Bright green
            'YELLOW': '\033[93m',    # Bright yellow
            'BLUE': '\033[94m',      # Bright blue
            'MAGENTA': '\033[95m',   # Bright magenta
            'RED': '\033[91m',       # Bright red
            'RESET': '\033[0m',      # Reset to default
            'BOLD': '\033[1m',       # Bold text
        }
        
        for activity in detected_activities:
            # Handle face presence separately to avoid over-counting
            if activity == 'face_presence':
                # Only log face detection once per cooldown period
                if not self.face_currently_detected:
                    self.face_currently_detected = True
                    self.last_face_detection_time = current_time
                    self.session_stats['face_detections'] = 1  # Always 1 (face present or not)
                    print(f"{COLORS['CYAN']}{COLORS['BOLD']}👤 Face detected at {current_time.strftime('%H:%M:%S')}{COLORS['RESET']}")
                continue
            
            # Check if this is a new activity (not detected in last 3 seconds)
            if activity not in self.last_activity_time or \
               (current_time - self.last_activity_time[activity]).seconds > 3:
                
                # Log new activity with visible colors
                if activity == 'drinking':
                    self.session_stats['drinking_count'] += 1
                    print(f"{COLORS['BLUE']}{COLORS['BOLD']}💧 Drinking water detected at {current_time.strftime('%H:%M:%S')}{COLORS['RESET']}")
                elif activity == 'eating':
                    self.session_stats['eating_count'] += 1
                    print(f"{COLORS['GREEN']}{COLORS['BOLD']}🍽️  Eating detected at {current_time.strftime('%H:%M:%S')}{COLORS['RESET']}")
                
                # Log to database
                confidence = self.detection_confidence.get(activity, 0)
                self.log_activity(
                    activity, 
                    self.condition_status, 
                    confidence,
                    np.mean(self.motion_history) if self.motion_history else 0,
                    {'detected_at': current_time.isoformat()}
                )
                
                self.activity_history.append({
                    'activity': activity,
                    'timestamp': current_time,
                    'confidence': confidence
                })
            
            self.last_activity_time[activity] = current_time

        # Update face detection status - if not detected, reset after cooldown
        if 'face_presence' not in detected_activities and self.face_currently_detected:
            if self.last_face_detection_time and \
               (current_time - self.last_face_detection_time).seconds > self.face_detection_cooldown:
                self.face_currently_detected = False
        
        self.current_activities = detected_activities
    
    def draw_enhanced_ui(self, frame, results, fps, motion_intensity):
        """Draw enhanced UI with unified color scheme - Yellow borders and titles"""
        h, w = frame.shape[:2]

        # Draw YOLO detections with smaller boxes
        annotated_frame = results.plot(line_width=1, font_size=0.4)
        overlay = annotated_frame.copy()

        # ==================== TOP LEFT: System Status ====================
        # Dark semi-transparent background with yellow border
        cv2.rectangle(overlay, (10, 10), (350, 190), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        cv2.rectangle(annotated_frame, (10, 10), (350, 190), (255, 255, 0), 2)  # Yellow border

        # Title - Yellow
        cv2.putText(annotated_frame, "PATIENT MONITOR", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow
        cv2.line(annotated_frame, (20, 43), (340, 43), (255, 255, 0), 2)  # Yellow line

        # FPS - Bright Green
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Bright green

        # Patient condition with enhanced color coding
        condition_colors = {
            'Good': (0, 255, 0),      # Bright green
            'Fair': (0, 255, 255),    # Cyan
            'Poor': (0, 165, 255),    # Orange
            'Unknown': (128, 128, 128) # Gray
        }
        condition_color = condition_colors.get(self.condition_status, (255, 255, 255))
        cv2.putText(annotated_frame, "Status:", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, self.condition_status, (100, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, condition_color, 2)

        # Motion intensity with enhanced bar
        cv2.putText(annotated_frame, "Motion Level:", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"{motion_intensity:.1f}%", (280, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)  # Yellow

        # Enhanced progress bar with gradient colors
        bar_width = int((motion_intensity / 100) * 200)
        cv2.rectangle(annotated_frame, (120, 110), (330, 130), (40, 40, 40), -1)  # Dark background
        cv2.rectangle(annotated_frame, (120, 110), (330, 130), (100, 100, 100), 2)  # Border

        # Color gradient based on intensity
        if motion_intensity > 30:
            bar_color = (0, 255, 0)  # Green - high activity
        elif motion_intensity > 10:
            bar_color = (0, 255, 255)  # Cyan - medium activity
        else:
            bar_color = (0, 165, 255)  # Orange - low activity

        cv2.rectangle(annotated_frame, (120, 110), (120 + bar_width, 130), bar_color, -1)

        # Alert indicator with pulsing effect
        if self.alert_triggered:
            cv2.rectangle(annotated_frame, (20, 140), (340, 170), (0, 0, 255), -1)  # Red background
            cv2.rectangle(annotated_frame, (20, 140), (340, 170), (255, 255, 255), 2)  # White border
            cv2.putText(annotated_frame, "! ALERT: LOW ACTIVITY DETECTED !", (28, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated_frame, (20, 140), (340, 170), (0, 100, 0), -1)  # Dark green
            cv2.rectangle(annotated_frame, (20, 140), (340, 170), (0, 255, 0), 2)  # Green border
            cv2.putText(annotated_frame, "System Status: NORMAL", (28, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
