from ultralytics import YOLO
import os

# Available YOLO11 models
AVAILABLE_MODELS = {
    'yolo11n.pt': {
        'name': 'YOLO11 Nano',
        'size': '5.2 MB',
        'speed': 'Fastest',
        'accuracy': 'Good',
        'mAP': '39.5%',
        'description': 'Best for real-time detection on CPU - RECOMMENDED',
        'recommended': True
    },
    'yolo11s.pt': {
        'name': 'YOLO11 Small',
        'size': '18.8 MB',
        'speed': 'Fast',
        'accuracy': 'Better',
        'mAP': '47.0%',
        'description': 'Balanced speed and accuracy'
    },
    'yolo11m.pt': {
        'name': 'YOLO11 Medium',
        'size': '41.8 MB',
        'speed': 'Moderate',
        'accuracy': 'Very Good',
        'mAP': '51.5%',
        'description': 'Good for GPU systems'
    },
    'yolo11l.pt': {
        'name': 'YOLO11 Large',
        'size': '52.9 MB',
        'speed': 'Slow',
        'accuracy': 'Excellent',
        'mAP': '53.4%',
        'description': 'High accuracy, needs GPU'
    },
    'yolo11x.pt': {
        'name': 'YOLO11 Extra Large',
        'size': '117.5 MB',
        'speed': 'Slowest',
        'accuracy': 'Best',
        'mAP': '54.7%',
        'description': 'Maximum accuracy, GPU required'
    }
}