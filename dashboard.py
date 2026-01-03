import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import time


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Patient Activity Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Manage database connections and queries"""
    
    def __init__(self, config):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.config)
            return True
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    def get_activity_logs(self, start_date=None, end_date=None, limit=500):
        """Fetch activity logs"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = "SELECT * FROM activity_logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching activity logs: {e}")
            return pd.DataFrame()
    
    def get_alerts(self, start_date=None, end_date=None, limit=100):
        """Fetch alerts"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()
    
    def get_session_stats(self, limit=20):
        """Fetch session statistics"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT * FROM session_stats
                ORDER BY session_start DESC
                LIMIT %s
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            return df
        except Exception as e:
            st.error(f"Error fetching session stats: {e}")
            return pd.DataFrame()
    
    def get_activity_summary(self, start_date=None, end_date=None):
        """Get activity summary statistics"""
        if not self.conn:
            return []
        
        try:
            query = """
                SELECT 
                    activity_type,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(motion_intensity) as avg_motion
                FROM activity_logs
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " GROUP BY activity_type"
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Exception as e:
            st.error(f"Error fetching summary: {e}")
            return []
        
    def get_hourly_activity(self, start_date=None, end_date=None):
        """Get hourly activity distribution"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    activity_type,
                    COUNT(*) as count,
                    AVG(motion_intensity) as avg_motion,
                    AVG(confidence_score) as avg_confidence
                FROM activity_logs
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " GROUP BY DATE_TRUNC('hour', timestamp), activity_type ORDER BY hour DESC"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching hourly activity: {e}")
            return pd.DataFrame()
    
    def get_condition_timeline(self, start_date=None, end_date=None):
        """Get patient condition over time"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    timestamp,
                    condition_status,
                    motion_intensity,
                    confidence_score
                FROM activity_logs
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching condition timeline: {e}")
            return pd.DataFrame()
    
    def get_motion_statistics(self, start_date=None, end_date=None):
        """Get motion intensity statistics"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    DATE_TRUNC('day', timestamp) as day,
                    AVG(motion_intensity) as avg_motion,
                    MAX(motion_intensity) as max_motion,
                    MIN(motion_intensity) as min_motion,
                    COUNT(*) as samples
                FROM activity_logs
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " GROUP BY DATE_TRUNC('day', timestamp) ORDER BY day DESC"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching motion statistics: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            

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
    """Main dashboard function"""
    
    st.markdown('<div class="main-header">🏥 Patient Activity Monitoring Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Connection options
        connection_mode = st.radio(
            "Connection Mode",
            ["Load from .env", "Manual Entry"],
            horizontal=True
        )
        
        if connection_mode == "Load from .env":
            if os.path.exists('.env'):
                db_config = get_db_config_from_env()
                st.success("✅ .env file loaded")
                with st.expander("View Configuration"):
                    st.write(f"Host: {db_config['host']}")
                    st.write(f"Port: {db_config['port']}")
                    st.write(f"Database: {db_config['database']}")
                    st.write(f"User: {db_config['user']}")
            else:
                st.warning("⚠️ .env file not found. Using manual entry.")
                connection_mode = "Manual Entry"
        
        if connection_mode == "Manual Entry":
            with st.expander("Database Settings", expanded=True):
                db_host = st.text_input("Host", "localhost")
                db_port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
                db_name = st.text_input("Database", "patient_monitoring")
                db_user = st.text_input("Username", "postgres")
                db_password = st.text_input("Password", type="password")
            
            db_config = {
                'host': db_host,
                'port': db_port,
                'database': db_name,
                'user': db_user,
                'password': db_password
            }
        
        connect_btn = st.button("🔌 Connect to Database", type="primary", width="stretch")
        
        st.markdown("---")
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        time_range = st.selectbox(
            "Select time range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
            key="time_range"
        )
        
        if time_range == "Custom":
            date_range = st.date_input(
                "Select range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                max_value=datetime.now()
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = datetime.now() - timedelta(days=7)
                end_date = datetime.now()
        else:
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days = days_map.get(time_range, 7)
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
        
        st.markdown("---")
        
        # Auto refresh
        st.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Interval (seconds)", 5, 60, 15)
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    
    if connect_btn or st.session_state.db_manager:
        if not st.session_state.db_manager:
            st.session_state.db_manager = DatabaseManager(db_config)
            
            if st.session_state.db_manager.connect():
                st.sidebar.success("✅ Connected to database")
            else:
                st.session_state.db_manager = None
                st.stop()
        
        db = st.session_state.db_manager
        
        # Fetch all data
        with st.spinner("Loading data..."):
            activity_logs = db.get_activity_logs(start_date, end_date)
            alerts = db.get_alerts(start_date, end_date)
            session_stats = db.get_session_stats()
            activity_summary = db.get_activity_summary(start_date, end_date)
            hourly_activity = db.get_hourly_activity(start_date, end_date)
            condition_timeline = db.get_condition_timeline(start_date, end_date)
            motion_stats = db.get_motion_statistics(start_date, end_date)
        
        # KEY METRICS - FIXED
        st.markdown("## 📊 Key Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate metrics properly
        if not activity_logs.empty:
            drinking_count = len(activity_logs[activity_logs['activity_type'] == 'drinking'])
            eating_count = len(activity_logs[activity_logs['activity_type'] == 'eating'])
            total_activities = len(activity_logs)
            avg_confidence = (activity_logs['confidence_score'].mean() * 100) if 'confidence_score' in activity_logs.columns else 0
        else:
            drinking_count = 0
            eating_count = 0
            total_activities = 0
            avg_confidence = 0
        
        alert_count = len(alerts) if not alerts.empty else 0
        
        with col1:
            st.metric("💧 Drinking Events", drinking_count)
        
        with col2:
            st.metric("🍽️ Eating Events", eating_count)
        
        with col3:
            st.metric("📋 Total Activities", total_activities)
        
        with col4:
            delta_value = None
            if alert_count > 0:
                delta_value = alert_count
            st.metric("🚨 Alerts", alert_count, delta=delta_value, delta_color="inverse")
        
        with col5:
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Debug information (optional - can be removed)
        with st.expander("🔍 Debug Info"):
            st.write(f"Total rows in activity_logs: {len(activity_logs)}")
            if not activity_logs.empty:
                st.write("Activity type distribution:")
                st.write(activity_logs['activity_type'].value_counts())
            else:
                st.write("No activity logs found")
        
        # PATIENT CONDITION STATUS
        st.markdown("## 🏥 Current Patient Status")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not activity_logs.empty:
                latest_activity = activity_logs.iloc[0]
                latest_condition = latest_activity['condition_status']
                latest_time = latest_activity['timestamp']
                
                if latest_condition == 'Good':
                    st.success(f"✅ Patient Condition: **{latest_condition}** (as of {latest_time})")
                elif latest_condition == 'Fair':
                    st.warning(f"⚠️ Patient Condition: **{latest_condition}** (as of {latest_time})")
                elif latest_condition == 'Poor':
                    st.error(f"❌ Patient Condition: **{latest_condition}** (as of {latest_time})")
                else:
                    st.info(f"ℹ️ Patient Condition: **{latest_condition}** (as of {latest_time})")
            else:
                st.info("ℹ️ No condition data available")
        
        with col2:
            if not alerts.empty:
                recent_alert = alerts.iloc[0]
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                severity_icon = severity_color.get(recent_alert['severity'], "⚪")
                st.markdown(f"""
                    <div class="alert-box">
                        <strong>{severity_icon} Latest Alert:</strong><br>
                        {recent_alert['message']}<br>
                        <small>{recent_alert['timestamp']}</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="success-box">
                        <strong>✅ No Recent Alerts</strong><br>
                        System operating normally
                    </div>
                """, unsafe_allow_html=True)
        