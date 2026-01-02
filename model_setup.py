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
    
    
def test_model(model_name='yolo11n.pt'):
    """
    Test a YOLO11 model with a sample detection
    """
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        model = YOLO(model_name)
        
        # Print model info
        print("\n📊 Model Information:")
        print(f"   Model: {AVAILABLE_MODELS[model_name]['name']}")
        print(f"   Size: {AVAILABLE_MODELS[model_name]['size']}")
        print(f"   mAP: {AVAILABLE_MODELS[model_name]['mAP']}")
        print(f"   Classes: {len(model.names)} objects (COCO dataset)")
        print(f"   Device: {model.device}")
        
        # Print relevant classes for our project
        print("\n🎯 Relevant Classes for Patient Monitoring:")
        relevant_classes = {
            0: 'person',
            39: 'bottle',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
        }
        
        print("\n   👤 Person Detection:")
        print(f"      • {relevant_classes[0]}")
        
        print("\n   💧 Drinking Items:")
        print(f"      • {relevant_classes[39]} (bottle)")
        print(f"      • {relevant_classes[41]} (cup)")
        
        print("\n   🍽️ Eating/Food Items:")
        for class_id in [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]:
            print(f"      • {relevant_classes[class_id]}")
        
        print("\n   🍴 Utensils:")
        for class_id in [42, 43, 44]:
            print(f"      • {relevant_classes[class_id]}")
        
        print(f"\n✅ Model {model_name} is ready to use!")
        print(f"   Total detectable objects: {len(model.names)}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False


def show_all_models():
    """Display all available YOLO11 models"""
    print("\n" + "=" * 90)
    print("📋 Available YOLO11 Models")
    print("=" * 90)
    
    print(YOLO11_IMPROVEMENTS)
    
    print("\n📊 Model Comparison Table:")
    print("-" * 90)
    print(f"{'Model':<20} {'Size':<12} {'Speed':<12} {'mAP':<10} {'Description':<40}")
    print("-" * 90)
    
    for model_file, info in AVAILABLE_MODELS.items():
        recommended = "⭐" if info.get('recommended') else "  "
        print(f"{recommended} {model_file:<18} {info['size']:<12} {info['speed']:<12} "
              f"{info['mAP']:<10} {info['description']:<40}")
    
    print("-" * 90)
    print("\n💡 Recommendation: Use yolo11n.pt for real-time patient monitoring")


def compare_with_yolov8():
    """Show comparison between YOLO11 and YOLOv8"""
    print("\n" + "=" * 90)
    print("⚖️  YOLO11 vs YOLOv8 Comparison")
    print("=" * 90)
    
    comparison = """
    
    Model Size Comparison (Nano versions):
    ┌─────────────┬──────────┬────────────┬──────────┐
    │ Model       │ Size     │ mAP        │ Speed    │
    ├─────────────┼──────────┼────────────┼──────────┤
    │ YOLOv8n     │ 6.3 MB   │ 37.3%      │ Baseline │
    │ YOLO11n ⭐  │ 5.2 MB   │ 39.5%      │ +12%     │
    └─────────────┴──────────┴────────────┴──────────┘
    
    Key Advantages of YOLO11:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✅ Smaller model size (5.2 MB vs 6.3 MB)
    ✅ Better accuracy (+2.2% mAP)
    ✅ Faster inference (+12% speed improvement)
    ✅ Better small object detection (important for food items)
    ✅ Lower memory consumption
    ✅ More efficient architecture
    
    Perfect for Patient Monitoring:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🎯 Better detection of small items (cups, utensils)
    ⚡ Faster real-time processing on CPU
    💾 Lower system resource usage
    🎨 Improved accuracy for eating/drinking detection
    """
    
    print(comparison)


def download_all_models():
    """Download all YOLO11 models (for comparison)"""
    print("\n📦 Downloading all YOLO11 models...")
    print("⚠️  This will download ~236 MB of data")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    success_count = 0
    for model_name in AVAILABLE_MODELS.keys():
        if download_model(model_name):
            success_count += 1
        print()
    
    print(f"\n✅ Successfully downloaded {success_count}/{len(AVAILABLE_MODELS)} models")


def setup_recommended_model():
    """Setup the recommended model for the project"""
    print("\n" + "=" * 90)
    print("🚀 Setting up YOLO11 for Patient Monitoring System")
    print("=" * 90)
    
    print("\n📦 Downloading YOLO11 Nano (yolo11n.pt)")
    print("   • Size: 5.2 MB")
    print("   • Optimized for real-time CPU detection")
    print("   • 39.5% mAP accuracy")
    print("   • 12% faster than YOLOv8\n")
    
    model = download_model('yolo11n.pt')
    if model:
        print("\n" + "=" * 90)
        test_model('yolo11n.pt')
        print("\n" + "=" * 90)
        print("✅ Setup complete! You can now run main_detection.py")
        print("=" * 90)
    else:
        print("\n❌ Setup failed. Please check your internet connection.")


def check_existing_models():
    """Check which models are already downloaded"""
    print("\n🔍 Checking for existing YOLO11 models...\n")
    
    found_models = []
    for model_name in AVAILABLE_MODELS.keys():
        if os.path.exists(model_name):
            size = os.path.getsize(model_name) / (1024 * 1024)  # Convert to MB
            print(f"✅ {model_name:<15} - {size:.1f} MB - {AVAILABLE_MODELS[model_name]['name']}")
            found_models.append(model_name)
        else:
            print(f"❌ {model_name:<15} - Not found")
    
    if found_models:
        print(f"\n✅ Found {len(found_models)} YOLO11 model(s)")
        return found_models
    else:
        print("\n⚠️  No YOLO11 models found. Run setup to download.")
        return []


def check_ultralytics_version():
    """Check if ultralytics supports YOLO11"""
    try:
        import ultralytics
        version = ultralytics.__version__
        print(f"\n📦 Ultralytics version: {version}")
        
        # YOLO11 requires ultralytics >= 8.3.0
        from packaging import version as pkg_version
        if pkg_version.parse(version) >= pkg_version.parse("8.3.0"):
            print("✅ YOLO11 is supported!")
            return True
        else:
            print("⚠️  YOLO11 requires ultralytics >= 8.3.0")
            print(f"   Current version: {version}")
            print("\n🔧 Update with: pip install --upgrade ultralytics")
            return False
    except ImportError:
        print("❌ Ultralytics not installed")
        print("   Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"⚠️  Could not check version: {e}")
        return False