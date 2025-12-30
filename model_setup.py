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

# YOLO11 Improvements over YOLOv8
YOLO11_IMPROVEMENTS = """
🎯 YOLO11 Key Improvements:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Better Accuracy: 2-3% mAP improvement over YOLOv8
⚡ Faster Speed: 10-15% faster inference
🎨 Improved Architecture: Enhanced C3k2 blocks
📊 Better Small Object Detection
🔍 Enhanced Feature Pyramid Network
💡 Lower Memory Consumption
🚀 Better Training Efficiency
"""


def download_model(model_name='yolo11n.pt'):
    """
    Download a YOLO11 model
    
    Args:
        model_name: Name of the model to download (default: yolo11n.pt)
    """
    print(f"\n📦 Downloading {model_name}...")
    print(f"   {AVAILABLE_MODELS[model_name]['description']}")
    
    try:
        model = YOLO(model_name)
        print(f"✅ Successfully downloaded {model_name}")
        print(f"   Location: {os.path.abspath(model_name)}")
        return model
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure ultralytics>=8.3.0 is installed")
        print("   3. Try: pip install --upgrade ultralytics")
        return None