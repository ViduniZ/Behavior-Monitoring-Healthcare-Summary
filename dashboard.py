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