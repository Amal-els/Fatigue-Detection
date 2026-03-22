# Driver Fatigue Monitor

A Streamlit dashboard for webcam-based driver fatigue monitoring with live alarms, runtime diagnostics, and a fallback path when MediaPipe is unavailable.

## Features

- Real-time webcam monitoring with `streamlit-webrtc`
- Multi-indicator fatigue detection with MediaPipe Tasks Face Landmarker:
  - Eye closure detection (Eye Aspect Ratio)
  - Yawn detection (Mouth Aspect Ratio)
  - Head pose analysis (pitch and yaw angles)
- OpenCV fallback mode so the app still runs without MediaPipe
- Smoothed EAR/MAR/head-pose tracking to reduce noisy alerts
- Composited fatigue score (55% eye, 25% yawn, 20% head pose)
- Live dashboard with safety state, lighting status, detector mode, and event history
- Stable diagnostics display without flickering
- Local looping alarm on Windows that triggers on any fatigue indicator

## Project Structure

- `fatigue_app.py`: Streamlit entrypoint
- `drowsy_detection.py`: detection pipeline and diagnostics
- `app_state.py`: thread-safe runtime dashboard state
- `alarm.py`: alarm playback abstraction
- `ui.py`: dashboard styles and reusable UI markup
- `models/face_landmarker_v2.task`: MediaPipe Tasks face landmark model

## Setup

### Quick Start (Recommended)

**Windows:**
```powershell
.\setup.bat
```

**Linux/macOS:**
```bash
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies from `requirements.txt`
- Download the MediaPipe face landmarker model
- Verify that everything is in place

### Manual Setup

If you prefer to set up manually:

1. Create a Python virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Download the MediaPipe model:
   - Download `face_landmarker_v2.task` from [MediaPipe Assets](https://storage.googleapis.com/mediapipe-tasks/vision/face_landmarker/v2/face_landmarker.task)
   - Place it in the `models/` directory

4. Run the app:
```powershell
streamlit run fatigue_app.py
```

### Using Pre-packaged Environment

If your local setup fails, this repository contains a working virtual environment on some Windows machines:
```powershell
.\fatigue_env\Scripts\python.exe -m streamlit run fatigue_app.py
```

## How It Works

The app monitors three key indicators to detect driver fatigue:

1. **Eye Closure**: Tracks the Eye Aspect Ratio (EAR) using eye landmarks
2. **Yawning**: Detects mouth opening using Mouth Aspect Ratio (MAR)
3. **Head Pose**: Monitors excessive head tilts in pitch and yaw angles

An alarm triggers when **any** of these indicators is detected for longer than the configured "Alarm Delay" threshold. Each indicator contributes to an overall fatigue score: 55% from eye closure, 25% from yawning, and 20% from poor head posture.

## Tuning

- `Alarm Delay`: how long any fatigue indicator must be present before the alarm triggers
- `Eye Threshold`: the eye-ratio threshold used to classify drowsiness (eye closure)
- `Smoothing Window`: the number of recent frames averaged together for all indicators
- `Low Light Threshold`: grayscale brightness threshold used for lighting feedback
- `Dashboard Refresh`: how often the right-hand status panel refreshes while streaming

## Suggested Next Steps

- Add tests for EAR/MAR calculations and drowsiness state transitions
- Replace the OpenCV fallback with a stronger lightweight landmark detector
- Log sessions to CSV or SQLite for later evaluation
- Add configurable thresholds for yawn and head-pose detection in the sidebar
