import cv2
import numpy as np
import os
import pickle

# ==============================
# CONFIG
# ==============================
CAMERA_INDEX = 0
DB_PATH = "parts_db"
DB_FILE = "features_db.pkl"
ORB_FEATURES = 2000

# ==============================
# INIT
# ==============================
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# ==============================
# PREPROCESS
# ==============================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(blur)

# ==============================
# SHAPE FEATURES
# ==============================
def extract_shape_features(gray):
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

    return np.hstack([area, perimeter, aspect_ratio, hu])

# ==============================
# MAIN
# ==============================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    database = {}

    print("\n=== PART DATABASE BUILDER ===")
    print("c : capture image")
    print("q : quit and save database\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = preprocess(frame)

        cv2.putText(frame, "Press 'c' to capture | 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow("Database Builder", frame)
        key = cv2.waitKey(1) & 0xFF

        # ================= CAPTURE =================
        if key == ord('c'):
            part_name = input("Enter part name: ").strip()

            if part_name == "":
                print("[WARN] Empty name. Skipped.")
                continue

            part_dir = os.path.join(DB_PATH, part_name)
            os.makedirs(part_dir, exist_ok=True)

            img_id = len(os.listdir(part_dir)) + 1
            img_path = os.path.join(part_dir, f"{img_id}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Image saved: {img_path}")

            kp, des = orb.detectAndCompute(gray, None)
            shape = extract_shape_features(gray)

            if des is None or shape is None:
                print("[WARN] Features not detected. Try better lighting.")
                continue

            if part_name not in database:
                database[part_name] = []

            database[part_name].append({
                "des": des,
                "shape": shape
            })

            print(f"[OK] Features added for '{part_name}'\n")

        # ================= QUIT =================
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ================= SAVE DB =================
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print("\n=== DATABASE SAVED ===")
    for part, samples in database.items():
        print(f"{part}: {len(samples)} samples")
    print("=====================\n")

if __name__ == "__main__":
    main()
