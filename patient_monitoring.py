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