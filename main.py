
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math
import time
import winsound
import threading

from database import init_database, save_alert

# ===== TAI MO HINH CNN =====
model = tf.keras.models.load_model("models/eye_cnn.keras")

# ===== MEDIAPIPE =====
mp_face_mesh = mp.solutions.face_mesh

# Landmark tinh EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Landmark cat anh mat cho CNN
LEFT_EYE_CROP = [
    33, 133, 159, 145, 153, 154,
    155, 157, 158, 160, 161, 163
]

RIGHT_EYE_CROP = [
    362, 263, 386, 374, 380, 381,
    382, 384, 385, 387, 388, 390
]

# Landmark mieng
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# ===== CAU HINH =====
CNN_THRESHOLD = 0.5
EAR_THRESHOLD = 0.21

DROWSY_SECONDS = 2.0
ALERT_INTERVAL = 3.0

MAR_THRESHOLD = 0.50
YAWN_SECONDS = 1.5
YAWN_ALERT_INTERVAL = 5.0

# ===== BIEN TRANG THAI =====
closed_start = None
yawn_start = None
last_alert = 0
last_yawn_alert = 0


def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


# ===== TINH EAR =====
def calculate_ear(landmarks, eye_points):
    p = [landmarks[i] for i in eye_points]

    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# ===== TINH MAR =====
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
        return 0

    return vertical / horizontal


# ===== CAT ANH MAT =====
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


# ===== CNN DU DOAN MAT =====
def predict_eyes(frame, landmarks):
    eyes = []

    for indices in [LEFT_EYE_CROP, RIGHT_EYE_CROP]:
        eye = crop_eye(frame, landmarks, indices)

        if eye is not None:
            eye_rgb = cv2.cvtColor(
                eye, cv2.COLOR_BGR2RGB
            )
            eyes.append(eye_rgb)

    if len(eyes) != 2:
        return None

    batch = np.array(eyes, dtype=np.float32)

    predictions = model.predict(
        batch, verbose=0
    ).flatten()

    # closed_eyes = 0, open_eyes = 1
    return float(np.mean(predictions))


# ===== AM THANH CANH BAO =====
def play_alert():
    try:
        print("DANG PHAT AM THANH CANH BAO!")
        winsound.Beep(1500, 1000)
    except Exception as e:
        print("LOI AM THANH:", e)


# ===== KHOI TAO DATABASE =====
init_database()

# ===== MO CAMERA =====
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Khong the mo webcam!")

print("He thong phat hien ngu gat CNN dang hoat dong...")
print("Nhan Q de thoat.")

try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = camera.read()

            if not ret:
                print("Khong doc duoc webcam!")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb)

            status = "NO FACE"
            color = (0, 0, 255)

            # ===== NEU NHAN DIEN DUOC KHUON MAT =====
            if results.multi_face_landmarks:
                landmarks = (
                    results.multi_face_landmarks[0].landmark
                )

                current_time = time.monotonic()

                # ===== TINH EAR =====
                left_ear = calculate_ear(
                    landmarks, LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE
                )

                ear = (left_ear + right_ear) / 2.0

                # ===== TINH MAR =====
                mar = calculate_mar(landmarks)

                # ===== CNN NHAN DIEN MAT =====
                open_probability = predict_eyes(
                    frame, landmarks
                )

                if open_probability is not None:

                    # Ket hop CNN va EAR
                    eyes_closed = (
                        open_probability < CNN_THRESHOLD
                        and ear < EAR_THRESHOLD
                    )

                    # ===== PHAT HIEN NGU GAT =====
                    if eyes_closed:
                        if closed_start is None:
                            closed_start = current_time

                        closed_duration = (
                            current_time - closed_start
                        )

                        if closed_duration >= DROWSY_SECONDS:
                            status = "DROWSY!"
                            color = (0, 0, 255)

                            if (
                                current_time - last_alert
                                >= ALERT_INTERVAL
                            ):
                                # Phat am thanh
                                print("PHAT HIEN NGU GAT - GOI HAM CANH BAO")
                                
                                threading.Thread(
                                    target=play_alert,
                                    daemon=True
                                ).start()

                                # Luu database
                                save_alert(
                                    "DROWSY",
                                    ear=ear,
                                    mar=mar
                                )

                                last_alert = current_time

                        else:
                            status = "CNN: EYES CLOSED"
                            color = (0, 165, 255)

                    else:
                        closed_start = None
                        status = "EYES OPEN"
                        color = (0, 255, 0)

                    # Hien thi xac suat CNN
                    cv2.putText(
                        frame,
                        f"CNN Open: {open_probability:.2f}",
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                else:
                    closed_start = None
                    status = "EYES NOT DETECTED"
                    color = (0, 165, 255)

                # ===== PHAT HIEN NGAP =====
                if mar > MAR_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = current_time

                    yawn_duration = (
                        current_time - yawn_start
                    )

                    if yawn_duration >= YAWN_SECONDS:
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
                            current_time - last_yawn_alert
                            >= YAWN_ALERT_INTERVAL
                        ):
                            save_alert(
                                "YAWNING",
                                ear=ear,
                                mar=mar
                            )

                            last_yawn_alert = current_time

                else:
                    yawn_start = None

                # ===== HIEN THI EAR =====
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # ===== HIEN THI MAR =====
                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # ===== VE LANDMARK MAT =====
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
                # Khong co khuon mat thi reset
                closed_start = None
                yawn_start = None

            # ===== HIEN THI TRANG THAI =====
            cv2.putText(
                frame,
                status,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            # ===== HIEN THI CAMERA =====
            cv2.imshow(
                "Driver Drowsiness Detection - CNN",
                frame
            )

            # Nhan Q de thoat
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
