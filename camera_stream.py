
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math
import time
import winsound
import threading

from database import init_database
from evidence_reports import save_alert_with_evidence as save_alert


# ================= CAU HINH =================

EAR_FALLBACK_SECONDS = 3.0
CNN_THRESHOLD = 0.052847
EAR_THRESHOLD = 0.21
DROWSY_SECONDS = 2.0
ALERT_INTERVAL = 3.0

MAR_THRESHOLD = 0.50
YAWN_SECONDS = 1.5
YAWN_ALERT_INTERVAL = 5.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CROP = [
    33, 133, 159, 145, 153, 154,
    155, 157, 158, 160, 161, 163
]

RIGHT_EYE_CROP = [
    362, 263, 386, 374, 380, 381,
    382, 384, 385, 387, 388, 390
]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ================= TINH TOAN =================

def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def calculate_ear(landmarks, points):
    p = [landmarks[i] for i in points]

    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2 * horizontal)


def calculate_mar(landmarks):
    vertical = distance(
        landmarks[MOUTH_TOP],
        landmarks[MOUTH_BOTTOM]
    )

    horizontal = distance(
        landmarks[MOUTH_LEFT],
        landmarks[MOUTH_RIGHT]
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]

    points = [
        (
            int(landmarks[i].x * w),
            int(landmarks[i].y * h)
        )
        for i in indices
    ]

    padding = 12

    x1 = max(0, min(p[0] for p in points) - padding)
    y1 = max(0, min(p[1] for p in points) - padding)
    x2 = min(w, max(p[0] for p in points) + padding)
    y2 = min(h, max(p[1] for p in points) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    eye = frame[y1:y2, x1:x2]

    if eye.size == 0:
        return None

    return cv2.resize(eye, (96, 96))


def play_alert():
    try:
        print("DANG PHAT AM THANH CANH BAO!")
        winsound.Beep(1500, 1000)
    except Exception as e:
        print("LOI AM THANH:", e)


# ================= CAMERA STREAM =================

class CameraStream:

    def __init__(self):
        # Theo doi FPS
        self.fps = 0.0
        self.fps_frame_count = 0
        self.fps_start_time = time.monotonic()

        # Camera
        self.camera = None
        self.running = False

        # Trang thai dashboard
        self.status_lock = threading.Lock()
        self.current_status = "CAMERA OFF"
        self.current_ear = None
        self.current_mar = None
        self.last_frame_time = 0

        # Tai mo hinh CNN
        self.model = tf.keras.models.load_model(
            "models/eye_cnn_v3.keras"
        )

        # Theo doi ngu gat
        self.closed_start = None
        self.ear_closed_start = None
        self.drowsy_logged = False

        # Theo doi ngap
        self.yawn_start = None

        # Lam muot ket qua CNN
        self.cnn_history = deque(maxlen=5)
        self.cnn_frame_counter = 0
        self.last_cnn_probability = None

        # Thoi gian canh bao
        self.last_alert = 0
        self.last_yawn_alert = 0

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh

        # Khoi tao database
        init_database()

    # ================= BAT CAMERA =================

    def start(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(0)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            raise RuntimeError("Khong mo duoc webcam!")

        self.running = True

        with self.status_lock:
            self.current_status = "STARTING"
            self.current_ear = None
            self.current_mar = None
            self.last_frame_time = 0

        print("Camera da khoi dong.")

    # ================= CNN NHAN DIEN MAT =================

    def predict_eyes(self, frame, landmarks):
        eyes = []

        for indices in [LEFT_EYE_CROP, RIGHT_EYE_CROP]:
            eye = crop_eye(frame, landmarks, indices)

            if eye is not None:
                eyes.append(
                    cv2.cvtColor(
                        eye,
                        cv2.COLOR_BGR2RGB
                    )
                )

        if len(eyes) != 2:
            return None

        batch = np.array(
            eyes,
            dtype=np.float32
        )

        predictions = self.model(
            batch,
            training=False
        ).numpy().flatten()

        return float(np.mean(predictions))

    # ================= XU LY KHUNG HINH =================

    def process_frame(self, frame, face_mesh):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        status = "NO FACE"
        color = (0, 0, 255)

        ear_value = None
        mar_value = None

        if results.multi_face_landmarks:
            landmarks = (
                results.multi_face_landmarks[0].landmark
            )

            current_time = time.monotonic()

            # Tinh EAR
            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (left_ear + right_ear) / 2.0

            # Tinh MAR
            mar = calculate_mar(landmarks)

            ear_value = float(ear)
            mar_value = float(mar)

            # CNN chay cach mot khung hinh
            self.cnn_frame_counter += 1

            if (
                self.cnn_frame_counter % 2 == 1
                or self.last_cnn_probability is None
            ):
                open_probability = self.predict_eyes(
                    frame,
                    landmarks
                )

                self.last_cnn_probability = open_probability
            else:
                open_probability = self.last_cnn_probability

            # Lam muot ket qua CNN
            if open_probability is not None:
                self.cnn_history.append(open_probability)

                smooth_probability = float(
                    np.median(self.cnn_history)
                )
            else:
                self.cnn_history.clear()
                smooth_probability = None

            # Phat hien ngu gat
            if open_probability is not None:
                cnn_closed = (
                    smooth_probability is not None
                    and smooth_probability < CNN_THRESHOLD
                )

                ear_closed = ear < EAR_THRESHOLD

                eyes_closed = cnn_closed and ear_closed

                # Theo doi CNN va EAR dong thoi
                if eyes_closed:
                    if self.closed_start is None:
                        self.closed_start = current_time
                else:
                    self.closed_start = None

                # Theo doi EAR du phong
                if ear_closed:
                    if self.ear_closed_start is None:
                        self.ear_closed_start = current_time
                else:
                    self.ear_closed_start = None

                cnn_duration = (
                    current_time - self.closed_start
                    if self.closed_start is not None
                    else 0.0
                )

                ear_duration = (
                    current_time - self.ear_closed_start
                    if self.ear_closed_start is not None
                    else 0.0
                )

                cnn_alert = cnn_duration >= DROWSY_SECONDS

                ear_fallback_alert = (
                    ear_duration >= EAR_FALLBACK_SECONDS
                )

                if cnn_alert or ear_fallback_alert:
                    status = "DROWSY!"
                    color = (0, 0, 255)

                    # Phat am thanh canh bao
                    if (
                        current_time - self.last_alert
                        >= ALERT_INTERVAL
                    ):
                        threading.Thread(
                            target=play_alert,
                            daemon=True
                        ).start()

                        self.last_alert = current_time

                    # Luu mot ban ghi cho moi dot ngu gat
                    if not self.drowsy_logged:
                        save_alert(
                            "DROWSY",
                            ear=ear,
                            mar=mar,
                            frame=frame.copy()
                        )

                        self.drowsy_logged = True

                elif ear_closed:
                    status = "CNN: EYES CLOSED"
                    color = (0, 165, 255)

                else:
                    status = "EYES OPEN"
                    color = (0, 255, 0)

                    self.drowsy_logged = False

            else:
                self.closed_start = None
                self.ear_closed_start = None
                self.cnn_history.clear()
                self.drowsy_logged = False

                status = "EYES NOT DETECTED"
                color = (0, 165, 255)

            # Phat hien ngap
            if mar > MAR_THRESHOLD:
                if self.yawn_start is None:
                    self.yawn_start = current_time

                duration = current_time - self.yawn_start

                if duration >= YAWN_SECONDS:
                    if status != "DROWSY!":
                        status = "YAWNING!"
                        color = (0, 165, 255)

                    cv2.putText(
                        frame,
                        "YAWNING!",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2
                    )

                    if (
                        current_time - self.last_yawn_alert
                        >= YAWN_ALERT_INTERVAL
                    ):
                        save_alert(
                            "YAWNING",
                            ear=ear,
                            mar=mar,
                            frame=frame.copy()
                        )

                        self.last_yawn_alert = current_time
            else:
                self.yawn_start = None

            # Hien thi EAR
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Hien thi MAR
            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Ve landmark quanh mat
            h, w = frame.shape[:2]

            for index in LEFT_EYE + RIGHT_EYE:
                point = landmarks[index]

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255, 255, 0),
                    -1
                )

        else:
            # Khong nhan dien duoc khuon mat
            self.closed_start = None
            self.ear_closed_start = None
            self.yawn_start = None
            self.cnn_history.clear()
            self.drowsy_logged = False

            # Khong dung lai ket qua CNN cu
            self.cnn_frame_counter = 0
            self.last_cnn_probability = None

        # Hien thi trang thai tren video
        cv2.putText(
            frame,
            status,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Cap nhat du lieu dashboard
        with self.status_lock:
            self.current_status = status
            self.current_ear = ear_value
            self.current_mar = mar_value
            self.last_frame_time = time.monotonic()

        return frame

    # ================= LUONG VIDEO =================

    def generate_frames(self):
        try:
            with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:

                while self.running:
                    if (
                        self.camera is None
                        or not self.camera.isOpened()
                    ):
                        print("Camera da bi ngat ket noi!")
                        break

                    success, frame = self.camera.read()

                    if not success or frame is None:
                        print("Khong doc duoc hinh anh tu webcam!")
                        break

                    # Lat anh webcam
                    frame = cv2.flip(frame, 1)

                    # Xu ly CNN, EAR, MAR
                    frame = self.process_frame(
                        frame,
                        face_mesh
                    )

                    # Tinh FPS
                    self.fps_frame_count += 1

                    elapsed = (
                        time.monotonic()
                        - self.fps_start_time
                    )

                    if elapsed >= 1.0:
                        self.fps = self.fps_frame_count / elapsed
                        self.fps_frame_count = 0
                        self.fps_start_time = time.monotonic()

                    # Hien thi FPS
                    cv2.putText(
                        frame,
                        f"FPS: {self.fps:.1f}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # Ma hoa JPEG
                    success, buffer = cv2.imencode(
                        ".jpg",
                        frame
                    )

                    if not success:
                        print("Khong ma hoa duoc khung hinh!")
                        continue

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )

        except GeneratorExit:
            # Trinh duyet dong ket noi video
            pass

        except Exception as e:
            print("LOI LUONG CAMERA:", e)

        finally:
            self.stop()

    # ================= API TRANG THAI =================

    def get_status(self):
        """
        Tra ve trang thai, EAR va MAR cho dashboard.
        Du lieu duoc doc an toan bang status_lock.
        """
        with self.status_lock:
            raw_status = self.current_status
            last_update = self.last_frame_time
            ear = self.current_ear
            mar = self.current_mar
            camera_running = self.running

        # Camera tat hoac khong co khung hinh moi
        if (
            not camera_running
            or last_update == 0
            or time.monotonic() - last_update > 5
        ):
            status = "STANDBY"
            ear = None
            mar = None

        else:
            status_map = {
                "DROWSY!": "DROWSY",
                "YAWNING!": "YAWNING",
                "EYES OPEN": "NORMAL",
                "NO FACE": "NO_FACE",
                "CNN: EYES CLOSED": "EYES_CLOSED",
                "EYES NOT DETECTED": "EYES_NOT_DETECTED",
            }

            status = status_map.get(
                raw_status,
                raw_status
            )

        return {
            "status": status,
            "ear": ear,
            "mar": mar,
            "raw_status": raw_status,
            "camera_on": bool(camera_running),
        }

    # ================= TAT CAMERA =================

    def stop(self):
        self.running = False

        # Reset bo dem ngu gat va ngap
        self.closed_start = None
        self.ear_closed_start = None
        self.yawn_start = None

        # Reset ket qua CNN
        self.cnn_frame_counter = 0
        self.last_cnn_probability = None
        self.cnn_history.clear()
        self.drowsy_logged = False

        # Reset dashboard
        with self.status_lock:
            self.current_status = "CAMERA OFF"
            self.current_ear = None
            self.current_mar = None
            self.last_frame_time = 0

        # Giai phong webcam
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        print("Camera da dung va duoc giai phong.")
