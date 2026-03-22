import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None

try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    MEDIAPIPE_TASKS_AVAILABLE = True
except Exception:
    BaseOptions = None
    FaceLandmarker = None
    FaceLandmarkerOptions = None
    VisionTaskRunningMode = None
    MEDIAPIPE_TASKS_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DetectionResult:
    play_alarm: bool
    detector_mode: str
    detector_detail: str
    eye_ratio: float
    smoothed_eye_ratio: float
    mouth_ratio: float
    head_pitch: float
    head_yaw: float
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


def resolve_face_landmarker_model_path():
    env_model_path = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    candidate_paths = [
        env_model_path,
        os.path.join(BASE_DIR, "models", "face_landmarker_v2.task"),
        os.path.join(BASE_DIR, "models", "face_landmarker.task"),
        os.path.join(BASE_DIR, "face_landmarker_v2.task"),
        os.path.join(BASE_DIR, "face_landmarker.task"),
    ]

    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def infer_mediapipe_status():
    if mp is None:
        return "MediaPipe package is not installed"
    if MEDIAPIPE_TASKS_AVAILABLE:
        model_path = resolve_face_landmarker_model_path()
        if model_path is None:
            return "MediaPipe Tasks is installed, but face_landmarker_v2.task is missing"
        return model_path
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return "Using legacy MediaPipe Solutions FaceMesh"
    return "MediaPipe face landmarks are unavailable in this environment"


def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    if mp is not None and MEDIAPIPE_TASKS_AVAILABLE:
        model_path = resolve_face_landmarker_model_path()
        if model_path is not None:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return FaceLandmarker.create_from_options(options), "MediaPipe Tasks", model_path

    if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return (
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ),
            "MediaPipe Solutions",
            "Legacy solutions API",
        )

    return None, "OpenCV Fallback", infer_mediapipe_status()


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


def get_landmark_point(landmarks, idx, frame_width, frame_height):
    landmark = landmarks[idx]
    return denormalize_coordinates(landmark.x, landmark.y, frame_width, frame_height)


def calculate_mouth_ratio(landmarks, mouth_idxs, image_w, image_h):
    top_lip = get_landmark_point(landmarks, mouth_idxs["top"], image_w, image_h)
    bottom_lip = get_landmark_point(landmarks, mouth_idxs["bottom"], image_w, image_h)
    left_mouth = get_landmark_point(landmarks, mouth_idxs["left"], image_w, image_h)
    right_mouth = get_landmark_point(landmarks, mouth_idxs["right"], image_w, image_h)

    if not all([top_lip, bottom_lip, left_mouth, right_mouth]):
        return 0.0, None

    mouth_height = distance(top_lip, bottom_lip)
    mouth_width = distance(left_mouth, right_mouth)
    if mouth_width == 0:
        return 0.0, None

    mouth_ratio = mouth_height / mouth_width
    return mouth_ratio, (top_lip, bottom_lip, left_mouth, right_mouth)


def calculate_head_pose(landmarks, pose_idxs, image_w, image_h):
    nose_tip = get_landmark_point(landmarks, pose_idxs["nose_tip"], image_w, image_h)
    chin = get_landmark_point(landmarks, pose_idxs["chin"], image_w, image_h)
    left_temple = get_landmark_point(landmarks, pose_idxs["left_temple"], image_w, image_h)
    right_temple = get_landmark_point(landmarks, pose_idxs["right_temple"], image_w, image_h)
    forehead = get_landmark_point(landmarks, pose_idxs["forehead"], image_w, image_h)

    if not all([nose_tip, chin, left_temple, right_temple, forehead]):
        return 0.0, 0.0, None

    face_center_x = (left_temple[0] + right_temple[0]) / 2.0
    face_width = max(1.0, distance(left_temple, right_temple))
    face_height = max(1.0, distance(forehead, chin))

    yaw = ((nose_tip[0] - face_center_x) / face_width) * 100.0
    vertical_center_y = (forehead[1] + chin[1]) / 2.0
    pitch = ((nose_tip[1] - vertical_center_y) / face_height) * 120.0

    return pitch, yaw, (nose_tip, chin, left_temple, right_temple, forehead)


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
        self.mouth_idxs = {
            "top": 13,
            "bottom": 14,
            "left": 78,
            "right": 308,
        }
        self.pose_idxs = {
            "nose_tip": 1,
            "chin": 152,
            "left_temple": 234,
            "right_temple": 454,
            "forehead": 10,
        }
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.yellow = (0, 220, 255)

        self.facemesh_model, self.mediapipe_backend, self.detector_detail = get_mediapipe_app()
        self.use_mediapipe = self.facemesh_model is not None

        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
        )

        self.smoothing_window = max(1, int(smoothing_window))
        self.ear_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.mouth_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.pitch_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.yaw_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "drowsy_time": 0.0,
            "color": self.green,
            "play_alarm": False,
        }
        self.ear_text_pos = (10, 30)
        self.video_timestamp_ms = 0

    def _reset_state(self):
        self.state_tracker["start_time"] = time.perf_counter()
        self.state_tracker["drowsy_time"] = 0.0
        self.state_tracker["color"] = self.green
        self.state_tracker["play_alarm"] = False

    def _build_result(
        self,
        eye_ratio: float,
        mouth_ratio: float,
        head_pitch: float,
        head_yaw: float,
        face_detected: bool,
        low_light: bool,
    ) -> DetectionResult:
        smoothed_ear = float(np.mean(self.ear_history)) if self.ear_history else 0.0
        smoothed_mouth = float(np.mean(self.mouth_history)) if self.mouth_history else 0.0
        smoothed_pitch = float(np.mean(self.pitch_history)) if self.pitch_history else 0.0
        smoothed_yaw = float(np.mean(self.yaw_history)) if self.yaw_history else 0.0

        yawn_signal = 0.0
        if face_detected:
            yawn_signal = max(0.0, min(1.0, (smoothed_mouth - 0.36) / 0.18))

        head_pose_signal = 0.0
        if face_detected:
            pose_offset = max(abs(smoothed_pitch), abs(smoothed_yaw))
            head_pose_signal = max(0.0, min(1.0, (pose_offset - 8.0) / 18.0))

        fatigue_score = 0.0
        if face_detected:
            eye_signal = max(0.0, min(1.0, self.state_tracker["drowsy_time"] / 3.0))
            fatigue_score = min(1.0, (0.55 * eye_signal) + (0.25 * yawn_signal) + (0.20 * head_pose_signal))

        return DetectionResult(
            play_alarm=self.state_tracker["play_alarm"],
            detector_mode=self.mediapipe_backend if self.use_mediapipe else "OpenCV Fallback",
            detector_detail=self.detector_detail,
            eye_ratio=eye_ratio,
            smoothed_eye_ratio=smoothed_ear,
            mouth_ratio=smoothed_mouth if face_detected else mouth_ratio,
            head_pitch=smoothed_pitch if face_detected else head_pitch,
            head_yaw=smoothed_yaw if face_detected else head_yaw,
            drowsy_time=float(self.state_tracker["drowsy_time"]),
            face_detected=face_detected,
            low_light=low_light,
            fatigue_score=fatigue_score,
        )

    def _update_drowsiness(self, smoothed_ratio: float, smoothed_mouth: float, smoothed_pitch: float, smoothed_yaw: float, thresholds: dict):
        # Check for multiple fatigue indicators
        is_closed_eyes = smoothed_ratio < thresholds["EAR_THRESH"]
        is_yawning = smoothed_mouth > 0.36
        is_bad_head_pose = max(abs(smoothed_pitch), abs(smoothed_yaw)) > 20.0
        
        # Trigger drowsiness if any single indicator is present
        if is_closed_eyes or is_yawning or is_bad_head_pose:
            end_time = time.perf_counter()
            self.state_tracker["drowsy_time"] += end_time - self.state_tracker["start_time"]
            self.state_tracker["start_time"] = end_time
            self.state_tracker["color"] = self.red
            self.state_tracker["play_alarm"] = self.state_tracker["drowsy_time"] >= thresholds["WAIT_TIME"]
        else:
            self._reset_state()

    def _annotate_frame(self, frame, eye_ratio: float, mouth_ratio: float, head_pitch: float, head_yaw: float):
        frame_h = frame.shape[0]
        mouth_pos = (10, 60)
        pose_pos = (10, 90)
        drowsy_time_pos = (10, int(frame_h * 0.84))
        alarm_pos = (10, int(frame_h * 0.92))
        color = self.state_tracker["color"]

        plot_text(frame, f"EAR: {eye_ratio:.2f}", self.ear_text_pos, color)
        plot_text(frame, f"MAR: {mouth_ratio:.2f}", mouth_pos, color)
        plot_text(frame, f"POSE P/Y: {head_pitch:.1f}/{head_yaw:.1f}", pose_pos, color)
        plot_text(frame, f"DROWSY: {self.state_tracker['drowsy_time']:.2f}s", drowsy_time_pos, color)
        if self.state_tracker["play_alarm"]:
            plot_text(frame, "WAKE UP!", alarm_pos, color)

    def _process_with_cascades(self, frame: np.ndarray, thresholds: dict):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_light = float(np.mean(gray)) < thresholds["LOW_LIGHT_THRESH"]
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            self.ear_history.clear()
            self.mouth_history.clear()
            self.pitch_history.clear()
            self.yaw_history.clear()
            self._reset_state()
            return frame, self._build_result(0.0, 0.0, 0.0, 0.0, False, low_light)

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        upper_face_gray = gray[y : y + h // 2, x : x + w]
        detected_eyes = self.eye_cascade.detectMultiScale(
            upper_face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )

        eye_ratios = [eh / ew for (_, _, ew, eh) in detected_eyes if ew > 0]
        eye_ratio = float(np.mean(eye_ratios)) if eye_ratios else 0.0
        self.ear_history.append(eye_ratio)
        smoothed_ratio = float(np.mean(self.ear_history)) if self.ear_history else eye_ratio
        self._update_drowsiness(smoothed_ratio, 0.0, 0.0, 0.0, thresholds)

        cv2.rectangle(frame, (x, y), (x + w, y + h), self.state_tracker["color"], 2)
        for ex, ey, ew, eh in detected_eyes[:2]:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), self.state_tracker["color"], 2)

        self._annotate_frame(frame, smoothed_ratio, 0.0, 0.0, 0.0)
        return frame, self._build_result(eye_ratio, 0.0, 0.0, 0.0, True, low_light)

    def _extract_landmarks(self, rgb_frame):
        if self.mediapipe_backend == "MediaPipe Tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.video_timestamp_ms += 33
            results = self.facemesh_model.detect_for_video(mp_image, self.video_timestamp_ms)
            return results.face_landmarks

        results = self.facemesh_model.process(rgb_frame)
        if not results.multi_face_landmarks:
            return []
        return [face.landmark for face in results.multi_face_landmarks]

    def _process_with_mediapipe(self, frame: np.ndarray, thresholds: dict):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_light = float(np.mean(gray_frame)) < thresholds["LOW_LIGHT_THRESH"]
        frame_h, frame_w, _ = frame.shape

        landmarks_list = self._extract_landmarks(rgb_frame)
        if not landmarks_list:
            self.ear_history.clear()
            self.mouth_history.clear()
            self.pitch_history.clear()
            self.yaw_history.clear()
            self._reset_state()
            return frame, self._build_result(0.0, 0.0, 0.0, 0.0, False, low_light)

        landmarks = landmarks_list[0]
        eye_ratio, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
        mouth_ratio, mouth_points = calculate_mouth_ratio(landmarks, self.mouth_idxs, frame_w, frame_h)
        head_pitch, head_yaw, pose_points = calculate_head_pose(landmarks, self.pose_idxs, frame_w, frame_h)
        self.ear_history.append(eye_ratio)
        self.mouth_history.append(mouth_ratio)
        self.pitch_history.append(head_pitch)
        self.yaw_history.append(head_yaw)
        smoothed_ratio = float(np.mean(self.ear_history))
        smoothed_mouth = float(np.mean(self.mouth_history))
        smoothed_pitch = float(np.mean(self.pitch_history))
        smoothed_yaw = float(np.mean(self.yaw_history))
        self._update_drowsiness(smoothed_ratio, smoothed_mouth, smoothed_pitch, smoothed_yaw, thresholds)

        frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["color"])
        if mouth_points:
            for point in mouth_points:
                cv2.circle(frame, point, 2, self.yellow, -1)
        if pose_points:
            for point in pose_points:
                cv2.circle(frame, point, 2, self.yellow, -1)
        self._annotate_frame(
            frame,
            smoothed_ratio,
            float(np.mean(self.mouth_history)),
            float(np.mean(self.pitch_history)),
            float(np.mean(self.yaw_history)),
        )
        return frame, self._build_result(eye_ratio, mouth_ratio, head_pitch, head_yaw, True, low_light)

    def process(self, frame: np.ndarray, thresholds: dict):
        frame.flags.writeable = False
        frame = frame.copy()
        frame.flags.writeable = True
        
        # Flip frame at the start for mirror effect with normal text
        frame = cv2.flip(frame, 1)

        if self.use_mediapipe:
            frame, detection_result = self._process_with_mediapipe(frame, thresholds)
        else:
            frame, detection_result = self._process_with_cascades(frame, thresholds)

        return frame, detection_result
