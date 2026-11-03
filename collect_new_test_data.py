
import cv2
import mediapipe as mp
import os
import time

# ===== CAU HINH =====
IMAGE_SIZE = (96, 96)
MAX_IMAGES = 20
SAVE_INTERVAL = 0.3

LEFT_EYE = [
    33, 133, 159, 145, 153, 154,
    155, 157, 158, 160, 161, 163
]

RIGHT_EYE = [
    362, 263, 386, 374, 380, 381,
    382, 384, 385, 387, 388, 390
]

mp_face_mesh = mp.solutions.face_mesh


def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]

    points = [
        (int(landmarks[i].x * w),
         int(landmarks[i].y * h))
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

    return cv2.resize(eye, IMAGE_SIZE)


def main():
    print("Chon trang thai mat:")
    print("1 - Mat mo (OPEN)")
    print("2 - Mat nham (CLOSED)")

    choice = input("Nhap 1 hoac 2: ").strip()

    if choice == "1":
        label = "open_eyes"
    elif choice == "2":
        label = "closed_eyes"
    else:
        print("Lua chon khong hop le!")
        return

    save_dir = os.path.join("data", "new_test", label)
    os.makedirs(save_dir, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Khong mo duoc webcam!")

    count = 0
    collecting = False
    last_save = 0

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while count < MAX_IMAGES:
                ret, frame = camera.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = face_mesh.process(rgb)

                if (
                    collecting
                    and results.multi_face_landmarks
                    and time.monotonic() - last_save >= SAVE_INTERVAL
                ):
                    landmarks = (
                        results.multi_face_landmarks[0].landmark
                    )

                    # Luu moi lan 1 anh mat, xen ke trai/phai
                    indices = (
                        LEFT_EYE if count % 2 == 0
                        else RIGHT_EYE
                    )

                    eye = crop_eye(frame, landmarks, indices)

                    if eye is not None:
                        filename = os.path.join(
                            save_dir,
                            f"test_{time.time_ns()}_{count}.jpg"
                        )

                        if cv2.imwrite(filename, eye):
                            count += 1
                            last_save = time.monotonic()

                cv2.putText(
                    frame,
                    f"{label}: {count}/{MAX_IMAGES}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "S: Start/Pause | Q: Quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Collect Test Data", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    collecting = not collecting
                    print(
                        "Dang thu thap" if collecting
                        else "Tam dung"
                    )

                elif key == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Da luu {count} anh vao {save_dir}")


if __name__ == "__main__":
    main()
