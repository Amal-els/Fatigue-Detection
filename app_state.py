from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import localtime, strftime
from typing import Deque, Dict


@dataclass
class RuntimeState:
    lock: Lock = field(default_factory=Lock)
    play_alarm: bool = False
    detector_mode: str = "Unavailable"
    detector_detail: str = ""
    eye_ratio: float = 0.0
    smoothed_eye_ratio: float = 0.0
    mouth_ratio: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    drowsy_time: float = 0.0
    fatigue_score: float = 0.0
    face_detected: bool = False
    low_light: bool = False
    last_face_detected: bool = False
    last_low_light: bool = False
    history: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=120))
    events: Deque[str] = field(default_factory=lambda: deque(maxlen=12))

    def update(self, result: dict):
        with self.lock:
            previous_alarm = self.play_alarm
            self.play_alarm = bool(result["play_alarm"])
            self.detector_mode = result["detector_mode"]
            self.detector_detail = result.get("detector_detail", "")
            self.eye_ratio = float(result["eye_ratio"])
            self.smoothed_eye_ratio = float(result["smoothed_eye_ratio"])
            self.mouth_ratio = float(result.get("mouth_ratio", 0.0))
            self.head_pitch = float(result.get("head_pitch", 0.0))
            self.head_yaw = float(result.get("head_yaw", 0.0))
            self.drowsy_time = float(result["drowsy_time"])
            self.fatigue_score = float(result["fatigue_score"])
            self.face_detected = bool(result["face_detected"])
            self.low_light = bool(result["low_light"])
            self.history.append(
                {
                    "EAR": self.smoothed_eye_ratio,
                    "Threshold": result.get("ear_threshold", 0.0),
                    "Fatigue Score": self.fatigue_score,
                }
            )

            if self.play_alarm and not previous_alarm:
                self.events.appendleft(f"{_format_time()} Alert triggered after {self.drowsy_time:.1f}s below threshold")
            elif previous_alarm and not self.play_alarm:
                self.events.appendleft(f"{_format_time()} Driver state returned to attentive")
            elif self.last_face_detected and not self.face_detected:
                self.events.appendleft(f"{_format_time()} Face not detected")
            elif not self.last_low_light and self.low_light:
                self.events.appendleft(f"{_format_time()} Lighting dropped below the guidance threshold")

            self.last_face_detected = self.face_detected
            self.last_low_light = self.low_light

    def snapshot(self):
        with self.lock:
            return {
                "play_alarm": self.play_alarm,
                "detector_mode": self.detector_mode,
                "detector_detail": self.detector_detail,
                "eye_ratio": self.eye_ratio,
                "smoothed_eye_ratio": self.smoothed_eye_ratio,
                "mouth_ratio": self.mouth_ratio,
                "head_pitch": self.head_pitch,
                "head_yaw": self.head_yaw,
                "drowsy_time": self.drowsy_time,
                "fatigue_score": self.fatigue_score,
                "face_detected": self.face_detected,
                "low_light": self.low_light,
                "history": list(self.history),
                "events": list(self.events),
            }


def _format_time() -> str:
    return f"[{strftime('%H:%M:%S', localtime())}]"
