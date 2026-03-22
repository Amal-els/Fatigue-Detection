import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional

import cv2
import numpy as np


import mediapipe as mp



@dataclass
class DetectionResult:
    play_alarm: bool
    detector_mode: str
    eye_ratio: float
    smoothed_eye_ratio: float
    drowsy_time: float
    face_detected: bool
    low_light: bool
    fatigue_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def denormalize_coordinates(normalized_x, normalized_y, image_width, image_height):
    if normalized_x is None or normalized_y is None:
        return None

    if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
        return None

    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return (x_px, y_px)


def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    return None


def distance(point_1, point_2):
    return sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        p2_p6 = distance(coords_points[1], coords_points[5])
        p3_p5 = distance(coords_points[2], coords_points[4])
        p1_p4 = distance(coords_points[0], coords_points[3])

        if p1_p4 == 0:
            return 0.0, None

        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    except Exception:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fnt_scale=0.75, thickness=2):
    return cv2.putText(image, text, origin, font, fnt_scale, color, thickness)


class VideoFrameHandler:
    def __init__(self, smoothing_window: int = 5):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.yellow = (0, 220, 255)

        self.facemesh_model = get_mediapipe_app()
        self.use_mediapipe = self.facemesh_model is not None

        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
        )

        self.smoothing_window = max(1, int(smoothing_window))
        self.ear_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "drowsy_time": 0.0,
            "color": self.green,
            "play_alarm": False,
        }
        self.ear_text_pos = (10, 30)

    def _reset_state(self):
        self.state_tracker["start_time"] = time.perf_counter()
        self.state_tracker["drowsy_time"] = 0.0
        self.state_tracker["color"] = self.green
        self.state_tracker["play_alarm"] = False

    def _build_result(self, eye_ratio: float, face_detected: bool, low_light: bool) -> DetectionResult:
        smoothed_ear = float(np.mean(self.ear_history)) if self.ear_history else 0.0
        fatigue_score = 0.0
        if face_detected:
            fatigue_score = max(0.0, min(1.0, self.state_tracker["drowsy_time"] / 3.0))

        return DetectionResult(
            play_alarm=self.state_tracker["play_alarm"],
            detector_mode="MediaPipe Face Mesh" if self.use_mediapipe else "OpenCV Fallback",
            eye_ratio=eye_ratio,
            smoothed_eye_ratio=smoothed_ear,
            drowsy_time=float(self.state_tracker["drowsy_time"]),
            face_detected=face_detected,
            low_light=low_light,
            fatigue_score=fatigue_score,
        )

    def _update_drowsiness(self, smoothed_ratio: float, thresholds: dict):
        if smoothed_ratio < thresholds["EAR_THRESH"]:
            end_time = time.perf_counter()
            self.state_tracker["drowsy_time"] += end_time - self.state_tracker["start_time"]
            self.state_tracker["start_time"] = end_time
            self.state_tracker["color"] = self.red
            self.state_tracker["play_alarm"] = self.state_tracker["drowsy_time"] >= thresholds["WAIT_TIME"]
        else:
            self._reset_state()

    def _annotate_frame(self, frame, eye_ratio: float):
        frame_h = frame.shape[0]
        drowsy_time_pos = (10, int(frame_h * 0.84))
        alarm_pos = (10, int(frame_h * 0.92))
        color = self.state_tracker["color"]

        plot_text(frame, f"EAR: {eye_ratio:.2f}", self.ear_text_pos, color)
        plot_text(frame, f"DROWSY: {self.state_tracker['drowsy_time']:.2f}s", drowsy_time_pos, color)
        if self.state_tracker["play_alarm"]:
            plot_text(frame, "WAKE UP!", alarm_pos, color)

    def _process_with_cascades(self, frame: np.ndarray, thresholds: dict):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_light = float(np.mean(gray)) < thresholds["LOW_LIGHT_THRESH"]
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            self.ear_history.clear()
            self._reset_state()
            return frame, self._build_result(0.0, False, low_light)

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        upper_face_gray = gray[y : y + h // 2, x : x + w]
        detected_eyes = self.eye_cascade.detectMultiScale(
            upper_face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )

        eye_ratios = [eh / ew for (_, _, ew, eh) in detected_eyes if ew > 0]
        eye_ratio = float(np.mean(eye_ratios)) if eye_ratios else 0.0
        self.ear_history.append(eye_ratio)
        smoothed_ratio = float(np.mean(self.ear_history)) if self.ear_history else eye_ratio
        self._update_drowsiness(smoothed_ratio, thresholds)

        cv2.rectangle(frame, (x, y), (x + w, y + h), self.state_tracker["color"], 2)
        for ex, ey, ew, eh in detected_eyes[:2]:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), self.state_tracker["color"], 2)

        self._annotate_frame(frame, smoothed_ratio)
        return frame, self._build_result(eye_ratio, True, low_light)

    def _process_with_mediapipe(self, frame: np.ndarray, thresholds: dict):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_light = float(np.mean(gray_frame)) < thresholds["LOW_LIGHT_THRESH"]
        frame_h, frame_w, _ = frame.shape

        results = self.facemesh_model.process(rgb_frame)
        if not results.multi_face_landmarks:
            self.ear_history.clear()
            self._reset_state()
            return frame, self._build_result(0.0, False, low_light)

        landmarks = results.multi_face_landmarks[0].landmark
        eye_ratio, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
        self.ear_history.append(eye_ratio)
        smoothed_ratio = float(np.mean(self.ear_history))
        self._update_drowsiness(smoothed_ratio, thresholds)

        frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["color"])
        self._annotate_frame(frame, smoothed_ratio)
        return frame, self._build_result(eye_ratio, True, low_light)

    def process(self, frame: np.ndarray, thresholds: dict):
        frame.flags.writeable = False
        frame = frame.copy()
        frame.flags.writeable = True

        if self.use_mediapipe:
            frame, detection_result = self._process_with_mediapipe(frame, thresholds)
        else:
            frame, detection_result = self._process_with_cascades(frame, thresholds)

        frame = cv2.flip(frame, 1)
        return frame, detection_result
