#!/usr/bin/env python
"""
Setup script for Driver Fatigue Monitor.
Installs dependencies and downloads the MediaPipe face landmarker model.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e}")
        return False


def create_venv():
    """Create a Python virtual environment if it doesn't exist."""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("\n📦 Creating virtual environment...")
    if sys.platform == "win32":
        return run_command(f"{sys.executable} -m venv .venv", "Create virtual environment")
    else:
        return run_command(f"python3 -m venv .venv", "Create virtual environment")


def get_pip_command():
    """Get the correct pip command for the platform."""
    if sys.platform == "win32":
        return ".venv\\Scripts\\pip.exe"
    else:
        return ".venv/bin/pip"


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    pip_cmd = get_pip_command()
    return run_command(f"{pip_cmd} install -r requirements.txt", "Install dependencies")


def download_model():
    """Download the MediaPipe face landmarker model."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "face_landmarker_v2.task"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    print("\n📥 Downloading MediaPipe face landmarker model...")
    model_url = "https://storage.googleapis.com/mediapipe-tasks/vision/face_landmarker/v2/face_landmarker.task"
    
    try:
        print(f"Downloading from: {model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model downloaded: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\n⚠️  You can manually download the model from:")
        print(f"   {model_url}")
        print(f"   And save it to: {model_path}")
        return False


def verify_setup():
    """Verify that all required files are in place."""
    print("\n" + "="*60)
    print("🔍 Verifying setup...")
    print("="*60)
    
    required_files = [
        ("fatigue_app.py", "Main application"),
        ("requirements.txt", "Dependencies"),
        ("models/face_landmarker_v2.task", "MediaPipe model"),
    ]
    
    all_ok = True
    for filepath, description in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"✅ {description}: {filepath}")
        else:
            print(f"❌ {description}: {filepath} - NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    """Run the complete setup."""
    print("\n" + "🚀 "*30)
    print("Driver Fatigue Monitor - Setup Script")
    print("🚀 "*30)
    
    steps = [
        ("Virtual Environment", create_venv),
        ("Dependencies", install_dependencies),
        ("MediaPipe Model", download_model),
        ("Verification", verify_setup),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n⚠️  Setup incomplete at step: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\n🎉 You can now run the app with:")
    if sys.platform == "win32":
        print("   .venv\\Scripts\\python.exe -m streamlit run fatigue_app.py")
    else:
        print("   source .venv/bin/activate")
        print("   streamlit run fatigue_app.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
