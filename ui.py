APP_CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(16, 185, 129, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.16), transparent 24%),
        linear-gradient(180deg, #08111f 0%, #0d1726 48%, #111827 100%);
    color: #e5eefb;
}
.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.02em;
}
.hero-card, .panel-card, .status-card {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(15, 23, 42, 0.78);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
}
.hero-card {
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.panel-card {
    padding: 1.15rem 1.2rem;
}
.status-card {
    padding: 1rem 1.15rem;
    margin-bottom: 0.9rem;
}
.eyebrow {
    color: #67e8f9;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.45rem;
}
.hero-copy, .muted-copy {
    color: #c7d2fe;
    line-height: 1.6;
}
.status-label {
    color: #94a3b8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.status-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
}
.status-value.alert {
    color: #fb7185;
}
.status-value.warn {
    color: #facc15;
}
.status-value.safe {
    color: #4ade80;
}
.event-item {
    color: #cbd5e1;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
div[data-testid="stSidebar"] {
    background: rgba(8, 15, 29, 0.94);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.8rem;
    border-radius: 18px;
}
</style>
"""


def render_hero():
    return """
    <div class="hero-card">
        <div class="eyebrow">Driver Safety Dashboard</div>
        <div class="hero-title">Real-time Fatigue Monitoring</div>
        <div class="hero-copy">
            Track eye activity from the live camera feed, smooth noisy detections, and raise a local
            alarm when drowsiness persists beyond your configured delay.
        </div>
    </div>
    """


def render_summary_cards(wait_time: float, ear_thresh: float, audio_mode: str):
    return f"""
    <div class="status-card">
        <div class="status-label">Alarm Delay</div>
        <div class="status-value">{wait_time:.2f}s</div>
    </div>
    <div class="status-card">
        <div class="status-label">Detection Threshold</div>
        <div class="status-value">{ear_thresh:.2f}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Audio Mode</div>
        <div class="status-value">{audio_mode}</div>
    </div>
    """


def render_safety_card(snapshot: dict):
    if snapshot["play_alarm"]:
        state_class = "alert"
        state_label = "Fatigue Alert"
    elif not snapshot["face_detected"]:
        state_class = "warn"
        state_label = "Face Not Found"
    elif snapshot["low_light"]:
        state_class = "warn"
        state_label = "Low Light"
    else:
        state_class = "safe"
        state_label = "Attentive"

    return f"""
    <div class="status-card">
        <div class="status-label">Safety State</div>
        <div class="status-value {state_class}">{state_label}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Detector</div>
        <div class="status-value">{snapshot['detector_mode']}</div>
    </div>
    <div class="status-card">
        <div class="status-label">Guidance</div>
        <div class="muted-copy">
            Keep your face centered, add front lighting if needed, and use the live chart to tune the threshold.
        </div>
    </div>
    """


def render_event_log(events: list[str]):
    if not events:
        return '<div class="muted-copy">No recent events yet. Start the stream to collect diagnostics.</div>'

    event_items = "".join([f'<div class="event-item">{event}</div>' for event in events[:6]])
    return event_items
