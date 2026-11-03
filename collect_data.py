
import cv2
import mediapipe as mp
import os
import time

# Thu muc luu anh
DATA_DIR = "data/glasses"

# Landmark bao quanh hai mat
LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 163]
RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 384, 385, 387, 388, 390]

# Moi lan chay thu thap toi da 200 anh
MAX_IMAGES = 100
SAVE_INTERVAL = 0.5

mp_face_mesh = mp.solutions.face_mesh


def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]

    points = [
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in indices
    ]

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    # Them le de anh khong bi cat sat mat
    padding = 12

    x1 = max(0, min(x_values) - padding)
    y1 = max(0, min(y_values) - padding)
    x2 = min(w, max(x_values) + padding)
    y2 = min(h, max(y_values) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    eye = frame[y1:y2, x1:x2]

    if eye.size == 0:
        return None

    return cv2.resize(eye, (96, 96))


def main():
    print("Chon trang thai mat:")
    print("1 - OPEN EYES (Mat mo)")
    print("2 - CLOSED EYES (Mat nham)")

    choice = input("Nhap 1 hoac 2: ").strip()

    if choice == "1":
        label = "open_eyes"
    elif choice == "2":
        label = "closed_eyes"
    else:
        print("Lua chon khong hop le!")
        return

    output_dir = os.path.join(DATA_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Khong mo duoc webcam!")
        return

    # Dem so anh da co trong thu muc
    count = len([
    filename for filename in os.listdir(output_dir)
    if filename.lower().endswith(".jpg")
])
    last_saved = 0

    print("Nhan S de bat dau thu thap.")
    print("Nhan Q de thoat.")

    collecting = False

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:
                ret, frame = camera.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                eyes = []

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    for indices in [LEFT_EYE, RIGHT_EYE]:
                        eye = crop_eye(frame, landmarks, indices)

                        if eye is not None:
                            eyes.append(eye)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    collecting = True
                    print("Bat dau thu thap anh...")

                if key == ord("q"):
                    break

                if collecting and eyes:
                    now = time.monotonic()

                    if now - last_saved >= SAVE_INTERVAL:
                        for eye in eyes:
                            if count >= MAX_IMAGES:
                                break

                            filename = os.path.join(
                                output_dir,
                                f"{label}_{time.time_ns()}.jpg"
                            )

                            cv2.imwrite(filename, eye)
                            count += 1

                        last_saved = now

                cv2.putText(
                    frame,
                    f"{label}: {count}/{MAX_IMAGES}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Collect Eye Dataset", frame)

                if count >= MAX_IMAGES:
                    print("Da thu thap du 200 anh!")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Da luu {count} anh vao {output_dir}")


if __name__ == "__main__":
    main()
