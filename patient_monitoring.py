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

        # Recording & Database status
        y_status = 182
        if self.is_recording:
            cv2.circle(annotated_frame, (28, y_status - 5), 7, (0, 0, 255), -1)  # Red dot
            cv2.putText(annotated_frame, "RECORDING", (42, y_status),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        db_status = "DB: CONNECTED" if self.db_conn else "DB: OFFLINE"
        db_color = (0, 255, 0) if self.db_conn else (128, 128, 128)
        cv2.putText(annotated_frame, db_status, (220, y_status),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, db_color, 2)

        # ==================== TOP RIGHT: Current Activities ====================
        cv2.rectangle(overlay, (w - 310, 10), (w - 10, 180), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        cv2.rectangle(annotated_frame, (w - 310, 10), (w - 10, 180), (255, 255, 0), 2)  # Yellow border

        cv2.putText(annotated_frame, "LIVE ACTIVITIES", (w - 300, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow
        cv2.line(annotated_frame, (w - 300, 43), (w - 20, 43), (255, 255, 0), 2)  # Yellow line

        y_offset = 70
        if self.current_activities:
            for activity in self.current_activities:
                if activity == 'face_presence':
                    icon = "👤"
                    text = "Face Detected"
                    color = (255, 200, 0)  # Light blue/cyan
                elif activity == 'drinking':
                    icon = "💧"
                    text = "Drinking Water"
                    color = (255, 150, 0)  # Blue
                elif activity == 'eating':
                    icon = "🍽"
                    text = "Eating Food"
                    color = (0, 255, 150)  # Green-cyan
                else:
                    icon = "•"
                    text = activity.title()
                    color = (200, 200, 200)

                confidence = self.detection_confidence.get(activity, 0) * 100

                # Activity name with icon
                cv2.putText(annotated_frame, f"{icon} {text}", (w - 290, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # Confidence percentage
                cv2.putText(annotated_frame, f"{confidence:.0f}%", (w - 80, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)  # Yellow

                # Mini confidence bar
                conf_bar_width = int((confidence / 100) * 50)
                bar_y = y_offset + 8
                cv2.rectangle(annotated_frame, (w - 85, bar_y), (w - 35, bar_y + 5), (60, 60, 60), -1)
                cv2.rectangle(annotated_frame, (w - 85, bar_y), (w - 85 + conf_bar_width, bar_y + 5), 
                            (0, 255, 255), -1)  # Cyan bar

                y_offset += 35
        else:
            cv2.putText(annotated_frame, "No activities detected", (w - 290, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            cv2.putText(annotated_frame, "Monitoring...", (w - 290, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

        # ==================== MIDDLE RIGHT: Session Statistics ====================
        cv2.rectangle(overlay, (w - 310, 195), (w - 10, 400), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        cv2.rectangle(annotated_frame, (w - 310, 195), (w - 10, 400), (255, 255, 0), 2)  # Yellow border

        cv2.putText(annotated_frame, "SESSION STATS", (w - 300, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow
        cv2.line(annotated_frame, (w - 300, 228), (w - 20, 228), (255, 255, 0), 2)  # Yellow line

        stats_items = [
            ("Drinking:", self.session_stats['drinking_count'], (100, 200, 255)),  # Light blue
            ("Eating:", self.session_stats['eating_count'], (100, 255, 150)),     # Light green
            ("Face Seen:", "YES" if self.session_stats['face_detections'] > 0 else "NO", 
             (255, 255, 0) if self.session_stats['face_detections'] > 0 else (128, 128, 128)),
            ("Alerts:", self.session_stats['alerts_triggered'], 
             (0, 0, 255) if self.session_stats['alerts_triggered'] > 0 else (0, 255, 0)),
        ]

        y_offset = 255
        for label, value, label_color in stats_items:
            # Label
            cv2.putText(annotated_frame, label, (w - 290, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

            # Value with background box
            value_str = str(value)
            cv2.rectangle(annotated_frame, (w - 90, y_offset - 18), (w - 30, y_offset + 2), 
                         (40, 40, 40), -1)
            cv2.putText(annotated_frame, value_str, (w - 80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow
            y_offset += 38

        # Overall condition with colored background
        cv2.putText(annotated_frame, "Overall Status:", (w - 290, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        condition_bg_color = {
            'Good': (0, 100, 0),
            'Fair': (0, 100, 100),
            'Poor': (0, 50, 100),
            'Unknown': (50, 50, 50)
        }
        bg_color = condition_bg_color.get(self.condition_status, (50, 50, 50))
        cv2.rectangle(annotated_frame, (w - 140, y_offset - 18), (w - 30, y_offset + 2), 
                     bg_color, -1)
        cv2.putText(annotated_frame, self.condition_status, (w - 130, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, condition_color, 2)

        # ==================== BOTTOM: Activity Timeline ====================
        timeline_height = 120
        cv2.rectangle(overlay, (10, h - timeline_height - 10), (w - 10, h - 10), (20, 20, 20), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        cv2.rectangle(annotated_frame, (10, h - timeline_height - 10), (w - 10, h - 10), 
                     (255, 255, 0), 2)  # Yellow border

        cv2.putText(annotated_frame, "ACTIVITY TIMELINE", (20, h - timeline_height + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow
        cv2.line(annotated_frame, (20, h - timeline_height + 23), (300, h - timeline_height + 23), 
                (255, 255, 0), 2)

        # Draw timeline
        if self.activity_history:
            x_start = 30
            x_spacing = min(60, (w - 60) // min(len(self.activity_history), 15))
            y_timeline = h - 55

            # Draw timeline base line
            cv2.line(annotated_frame, (x_start, y_timeline), (w - 30, y_timeline), 
                    (100, 100, 100), 2)

            # Show last 15 activities
            recent_activities = list(self.activity_history)[-15:]

            for i, activity_info in enumerate(recent_activities):
                x_pos = x_start + (i * x_spacing)
                activity = activity_info['activity']
                timestamp = activity_info['timestamp']

                # Activity icon and color
                if activity == 'drinking':
                    color = (255, 200, 0)  # Light blue
                    icon = "💧"
                elif activity == 'eating':
                    color = (0, 255, 150)  # Light green
                    icon = "🍽"
                else:
                    color = (150, 150, 150)
                    icon = "•"

                # Draw vertical line
                cv2.line(annotated_frame, (x_pos, y_timeline - 10), (x_pos, y_timeline + 10), 
                        color, 2)

                # Draw point with glow effect
                cv2.circle(annotated_frame, (x_pos, y_timeline), 8, (0, 0, 0), -1)  # Shadow
                cv2.circle(annotated_frame, (x_pos, y_timeline), 6, color, -1)
                cv2.circle(annotated_frame, (x_pos, y_timeline), 6, (255, 255, 255), 1)  # White outline

                # Time label
                time_str = timestamp.strftime('%H:%M')
                cv2.putText(annotated_frame, time_str, (x_pos - 22, y_timeline + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                # Activity label
                activity_label = activity[:3].upper()
                cv2.putText(annotated_frame, activity_label, (x_pos - 15, y_timeline - 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        else:
            cv2.putText(annotated_frame, "No activities recorded yet - monitoring in progress...", 
                       (w // 2 - 220, h - 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Timestamp with enhanced styling
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(annotated_frame, (w - 200, h - 30), (w - 15, h - 12), (40, 40, 40), -1)
        cv2.putText(annotated_frame, timestamp, (w - 195, h - 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # Cyan

        return annotated_frame
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_monitoring_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20
            frame_size = (frame.shape[1], frame.shape[0])
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            self.is_recording = True
            print(f"🔴 Recording started: {filename}")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            print("⏹️  Recording stopped")
    
    def process_frame(self, frame):
        """Process frame with all detections"""
        start_time = time.time()
        
        # Detect motion
        _, motion_intensity, motion_areas, has_significant_motion = self.detect_motion(frame)
        
        # Detect activities
        results, detected_activities, face_detected = self.detect_activities(frame)
        
        # Assess patient condition
        condition = self.assess_patient_condition(motion_intensity, face_detected, has_significant_motion)
        
        # Process activities
        self.process_activities(detected_activities)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1 / elapsed if elapsed > 0 else 0
        self.fps_counter.append(fps)
        avg_fps = np.mean(self.fps_counter)
        
        # Draw enhanced UI
        annotated_frame = self.draw_enhanced_ui(frame, results, avg_fps, motion_intensity)
        
        # Record if enabled
        if self.is_recording and self.video_writer:
            self.video_writer.write(annotated_frame)
        
        return annotated_frame
    
    def run(self, camera_source=0):
        """Main monitoring loop"""
        print("\n" + "=" * 80)
        print("🏥 PATIENT ACTIVITY MONITORING SYSTEM - ACTIVE")
        print("=" * 80)
        print("\n📹 Opening camera...")
        
        # Support for IP camera
        if isinstance(camera_source, str) and camera_source.startswith('rtsp'):
            print(f"🌐 Connecting to IP camera: {camera_source}")
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera opened successfully!")
        print("\n📋 Controls:")
        print("   Q - Quit")
        print("   R - Toggle Recording")
        print("   S - Save Screenshot")
        print("   A - Manual Alert Test")
        print("\n🎬 Monitoring started...\n")
        
        session_start = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Patient Activity Monitor', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Stopping monitoring...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.toggle_recording(annotated_frame)
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('a') or key == ord('A'):
                    self.trigger_alert("Manual Test", "Medium", "Manual alert triggered by user")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            
            if self.is_recording and self.video_writer:
                self.video_writer.release()
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Log session statistics
            if self.db_conn:
                try:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        INSERT INTO session_stats 
                        (session_start, session_end, total_activities, drinking_count, 
                         eating_count, motion_events, alerts_count, average_condition)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        session_start, datetime.now(),
                        len(self.activity_history),
                        self.session_stats['drinking_count'],
                        self.session_stats['eating_count'],
                        self.session_stats['motion_events'],
                        self.session_stats['alerts_triggered'],
                        self.condition_status
                    ))
                    self.db_conn.commit()
                    cursor.close()
                except Exception as e:
                    print(f"⚠️  Error saving session stats: {e}")
            
            if self.db_conn:
                self.db_conn.close()
     
            # Print final statistics
            print("\n" + "=" * 80)
            print("📊 SESSION SUMMARY")
            print("=" * 80)
            print(f"Session Duration: {datetime.now() - session_start}")
            print(f"Average FPS: {np.mean(self.fps_counter):.1f}")
            print(f"\n📋 Activities Detected:")
            print(f"   💧 Drinking Events: {self.session_stats['drinking_count']}")
            print(f"   🍽  Eating Events: {self.session_stats['eating_count']}")
            print(f"   👤 Face Detections: {self.session_stats['face_detections']}")
            print(f"   🚨 Alerts Triggered: {self.session_stats['alerts_triggered']}")
            print(f"\n🏥 Final Patient Condition: {self.condition_status}")
            print("=" * 80)


def get_db_config_from_env():
    """Load database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'patient_monitoring'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }


def main():
    """Main function"""
    print("=" * 80)
    print("🏥 PATIENT ACTIVITY MONITORING SYSTEM")
    print("   Using YOLO11 + OpenCV + PostgreSQL")
    print("=" * 80)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("\n✅ .env file found!")
        print("📖 Loading configuration from .env file...")
    else:
        print("\n⚠️  .env file not found!")
        print("📄 Creating .env file template...")
        create_env_template()
    
    print("\n⚙️  System Configuration:")
    print("   1. Quick Start (Load from .env)")
    print("   2. Custom Configuration")
    print("   3. IP Camera Setup")
    print("   4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip() or '1'
    
    # Default settings
    model_name = 'yolo11n.pt'
    confidence = 0.6
    camera_source = 0
    db_config = None
    
    if choice == '1':
        # Load from .env file
        print("\n📖 Loading configuration from .env file...")
        try:
            db_config = get_db_config_from_env()
            print("✅ Configuration loaded successfully!")
            print(f"   Host: {db_config['host']}")
            print(f"   Database: {db_config['database']}")
            print(f"   User: {db_config['user']}")
        except Exception as e:
            print(f"❌ Error loading .env: {e}")
            db_config = None
    
    elif choice == '2':
        # Custom configuration
        print("\n📦 YOLO Model Selection:")
        print("   1. Nano (Fastest, ~5 MB) - Recommended for real-time")
        print("   2. Small (Balanced, ~18 MB)")
        print("   3. Medium (Most Accurate, ~41 MB)")
        
        model_choice = input("Select model (1-3): ").strip() or '1'
        model_options = {'1': 'yolo11n.pt', '2': 'yolo11s.pt', '3': 'yolo11m.pt'}
        model_name = model_options.get(model_choice, 'yolo11n.pt')
        
        confidence_input = input("Confidence threshold (0.3-0.9, default 0.6): ").strip()
        confidence = float(confidence_input) if confidence_input else 0.6
        
        camera_input = input("Camera index (default 0): ").strip()
        camera_source = int(camera_input) if camera_input else 0
        
        # Database configuration
        print("\n💾 Database Configuration:")
        use_db = input("Enable PostgreSQL logging? (y/n): ").strip().lower()
        
        if use_db == 'y':
            print("\nEnter database credentials:")
            db_config = {
                'host': input("  Host (default: localhost): ").strip() or 'localhost',
                'port': int(input("  Port (default: 5432): ").strip() or 5432),
                'database': input("  Database name: ").strip(),
                'user': input("  Username: ").strip(),
                'password': input("  Password: ").strip()
            }
    
    elif choice == '3':
        # IP Camera setup
        print("\n🌐 IP Camera Configuration:")
        print("   Examples:")
        print("   • RTSP: rtsp://username:password@192.168.1.100:554/stream")
        print("   • HTTP: http://192.168.1.100:8080/video")
        
        camera_source = input("\nEnter camera URL: ").strip()
        
        confidence_input = input("Confidence threshold (default 0.6): ").strip()
        confidence = float(confidence_input) if confidence_input else 0.6
        
        # Database configuration
        use_db = input("\nEnable PostgreSQL logging? (y/n): ").strip().lower()
        if use_db == 'y':
            print("\nChoose database configuration source:")
            print("   1. Load from .env file")
            print("   2. Enter manually")
            db_choice = input("Select (1-2): ").strip() or '1'
            
            if db_choice == '1':
                db_config = get_db_config_from_env()
            else:# Change with your DB credentials
                db_config = {
                    'host': input("  Host (default: localhost): ").strip() or 'localhost',
                    'port': int(input("  Port (default: 5432): ").strip() or 5432),
                    'database': input("  Database name: ").strip() or 'patient_monitoring',
                    'user': input("  Username: ").strip() or 'postgres',
                    'password': input("  Password: ").strip() or '19990806',
                }
    
    elif choice == '4':
        print("👋 Exiting...")
        return
    
    else:
        print("❌ Invalid choice. Using default settings...")
    
    # Create and run monitor
    print("\n" + "=" * 80)
    print("🚀 Initializing System...")
    print("=" * 80)
    
    # Test database connection if configured
    if db_config:
        print(f"\n📊 Testing database connection...")
        try:
            test_conn = psycopg2.connect(**db_config)
            print("✅ Database connection test successful!")
            test_conn.close()
        except psycopg2.OperationalError as e:
            print(f"❌ Database connection test failed!")
            print(f"   Error: {e}")
            retry = input("\n   Continue without database? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting...")
                return
            db_config = None
        except Exception as e:
            print(f"❌ Unexpected database error: {e}")
            db_config = None
    
    monitor = PatientActivityMonitor(
        model_name=model_name,
        confidence=confidence,
        db_config=db_config
    )
    
    monitor.run(camera_source=camera_source)


def create_env_template():
    """Create a .env template file"""
    env_template = """# Patient Activity Monitoring System - Database Configuration
# Copy this template to .env and fill in your database credentials

# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=patient_monitoring
DB_USER=postgres
DB_PASSWORD=your_password_here

# Optional: Additional Configuration
# YOLO_MODEL=yolo11n.pt
# CONFIDENCE_THRESHOLD=0.6
# CAMERA_SOURCE=0
"""
    
    try:
        with open('.env.example', 'w') as f:
            f.write(env_template)
        print("\n📄 Template file created: .env.example")
        print("   Copy this file to .env and fill in your database credentials")
    except Exception as e:
        print(f"⚠️  Could not create template file: {e}")


if __name__ == "__main__":
    main()