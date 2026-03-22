import os
import time

import av
import streamlit as st
from config import VideoHTMLAttributes

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = exc

from alarm import AlarmPlayer
from app_state import RuntimeState
from drowsy_detection import VideoFrameHandler
from ui import APP_CSS, render_event_log, render_hero, render_safety_card, render_summary_cards

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_FILE_PATH = os.path.join(BASE_DIR, "audio", "wake_up.wav")


def ensure_session_defaults():
    defaults = {
        "wait_time": 1.0,
        "ear_thresh": 0.18,
        "smoothing_window": 5,
        "low_light_thresh": 55,
        "use_stun_server": False,
        "refresh_ms": 250,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "runtime_state" not in st.session_state:
        st.session_state.runtime_state = RuntimeState()
    if "alarm_player" not in st.session_state:
        st.session_state.alarm_player = AlarmPlayer(ALARM_FILE_PATH)


def render_missing_dependency_message():
    st.error("Missing dependency: `streamlit-webrtc` is not installed in the active Python environment.")
    st.info(
        "Run the app with `fatigue_env\\Scripts\\python.exe -m streamlit run fatigue_app.py` "
        "or install the project dependencies into the environment you want to use."
    )
    st.code(".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", language="powershell")
    st.stop()


def render_sidebar():
    st.sidebar.markdown("## Monitoring Controls")
    st.sidebar.caption("Tune detection sensitivity, smoothing, and stream behavior for your session.")
    wait_time = st.sidebar.slider("Alarm Delay (seconds)", 0.0, 5.0, key="wait_time", step=0.25)
    ear_thresh = st.sidebar.slider("Eye Threshold", 0.0, 0.4, key="ear_thresh", step=0.01)
    smoothing_window = st.sidebar.slider("Smoothing Window", 1, 15, key="smoothing_window")
    low_light_thresh = st.sidebar.slider("Low Light Threshold", 20, 120, key="low_light_thresh")
    refresh_ms = st.sidebar.slider("Dashboard Refresh (ms)", 100, 1000, key="refresh_ms", step=50)
    use_stun_server = st.sidebar.toggle(
        "Use STUN server",
        key="use_stun_server",
        help="Enable only when you need remote WebRTC connectivity beyond the local machine or network.",
    )

    st.sidebar.markdown("## Runtime Notes")
    st.sidebar.caption("MediaPipe gives better landmarks when installed. The OpenCV fallback keeps the demo usable.")

    return {
        "EAR_THRESH": ear_thresh,
        "WAIT_TIME": wait_time,
        "LOW_LIGHT_THRESH": low_light_thresh,
        "SMOOTHING_WINDOW": smoothing_window,
        "REFRESH_MS": refresh_ms,
        "USE_STUN_SERVER": use_stun_server,
    }


def render_system_status(runtime_state: RuntimeState, metrics_placeholder, status_placeholder, events_placeholder, is_streaming: bool):
    snapshot = runtime_state.snapshot()

    with metrics_placeholder.container():
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Stream", "Live" if is_streaming else "Idle")
        with metric_col2:
            st.metric("Alarm", "Active" if snapshot["play_alarm"] else "Standby")
        with metric_col3:
            st.metric("Face", "Detected" if snapshot["face_detected"] else "Searching")
        with metric_col4:
            st.metric("Lighting", "Low" if snapshot["low_light"] else "Good")

    with status_placeholder.container():
        left_col, right_col = st.columns([0.95, 1.45], gap="large")
        with left_col:
            st.markdown(render_safety_card(snapshot), unsafe_allow_html=True)
        with right_col:
            st.caption("Live diagnostics")
            history = snapshot["history"]
            if history:
                chart_data = {
                    "EAR": [item["EAR"] for item in history],
                    "Threshold": [item["Threshold"] for item in history],
                    "Fatigue Score": [item["Fatigue Score"] for item in history],
                }
                st.line_chart(chart_data, height=260)
            else:
                st.info("Start the stream to populate the EAR and fatigue trend.")

            detail_col1, detail_col2, detail_col3 = st.columns(3)
            detail_col1.metric("Raw EAR", f"{snapshot['eye_ratio']:.2f}")
            detail_col2.metric("Smoothed EAR", f"{snapshot['smoothed_eye_ratio']:.2f}")
            detail_col3.metric("Fatigue Score", f"{snapshot['fatigue_score']:.2f}")

    with events_placeholder.container():
        st.caption("Recent events")
        st.markdown(render_event_log(snapshot["events"]), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Driver Fatigue Monitor",
        page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Real-time fatigue monitoring with webcam-based eye analysis."},
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    ensure_session_defaults()

    if WEBRTC_IMPORT_ERROR is not None:
        render_missing_dependency_message()

    runtime_state: RuntimeState = st.session_state.runtime_state
    alarm_player: AlarmPlayer = st.session_state.alarm_player
    thresholds = render_sidebar()
    video_handler = VideoFrameHandler(smoothing_window=thresholds["SMOOTHING_WINDOW"])

    rtc_configuration = None
    if thresholds["USE_STUN_SERVER"]:
        rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    hero_col, status_col = st.columns([1.9, 1], gap="large")
    with hero_col:
        st.markdown(render_hero(), unsafe_allow_html=True)
    with status_col:
        st.markdown(
            render_summary_cards(
                thresholds["WAIT_TIME"],
                thresholds["EAR_THRESH"],
                alarm_player.audio_mode,
            ),
            unsafe_allow_html=True,
        )

    def video_frame_callback(frame: av.VideoFrame):
        image = frame.to_ndarray(format="bgr24")
        image, detection_result = video_handler.process(image, thresholds)
        result_payload = detection_result.to_dict()
        result_payload["ear_threshold"] = thresholds["EAR_THRESH"]
        runtime_state.update(result_payload)
        alarm_player.update(detection_result.play_alarm)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Live Camera Feed")
    st.caption("The alarm plays on this computer when prolonged eye closure is detected.")
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": {"height": {"ideal": 540}}, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    is_streaming = bool(ctx and ctx.state.playing)
    if not is_streaming:
        alarm_player.stop()

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Dashboard and Diagnostics")
    st.caption("Expanded runtime status appears below the stream for easier monitoring.")
    metrics_placeholder = st.empty()
    status_placeholder = st.empty()
    events_placeholder = st.empty()
    render_system_status(
        runtime_state,
        metrics_placeholder,
        status_placeholder,
        events_placeholder,
        is_streaming,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if is_streaming:
        while ctx.state.playing:
            render_system_status(
                runtime_state,
                metrics_placeholder,
                status_placeholder,
                events_placeholder,
                True,
            )
            time.sleep(thresholds["REFRESH_MS"] / 1000)


if __name__ == "__main__":
    main()
