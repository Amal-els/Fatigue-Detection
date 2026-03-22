# Driver Fatigue Monitor

A Streamlit dashboard for webcam-based driver fatigue monitoring with live alarms, runtime diagnostics, and a fallback path when MediaPipe is unavailable.

## Features

- Real-time webcam monitoring with `streamlit-webrtc`
- Eye-based fatigue detection with MediaPipe Face Mesh when available
- OpenCV fallback mode so the app still runs without MediaPipe
- Smoothed EAR tracking to reduce noisy alerts
- Live dashboard with safety state, lighting status, detector mode, and event history
- Local looping alarm on Windows

## Project Structure

- `fatigue_app.py`: Streamlit entrypoint
- `drowsy_detection.py`: detection pipeline and diagnostics
- `app_state.py`: thread-safe runtime dashboard state
- `alarm.py`: alarm playback abstraction
- `ui.py`: dashboard styles and reusable UI markup

## Setup

1. Create or activate a Python virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run fatigue_app.py
```

If your local `.venv` is broken on Windows, this repository already contains a working environment on some machines:

```powershell
.\fatigue_env\Scripts\python.exe -m streamlit run fatigue_app.py
```

## Tuning

- `Alarm Delay`: how long the smoothed eye ratio must stay below the threshold before the alarm triggers
- `Eye Threshold`: the eye-ratio threshold used to classify drowsiness
- `Smoothing Window`: the number of recent frames averaged together
- `Low Light Threshold`: grayscale brightness threshold used for lighting feedback
- `Dashboard Refresh`: how often the right-hand status panel refreshes while streaming

## Suggested Next Steps

- Add tests for EAR calculations and drowsiness state transitions
- Replace the OpenCV fallback with a stronger lightweight landmark detector
- Log sessions to CSV or SQLite for later evaluation
- Add yawn and head-pose signals for a more reliable fatigue score
