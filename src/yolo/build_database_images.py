import cv2
import numpy as np
import os
import pickle

# ==============================
# CONFIG
# ==============================
DB_PATH = "parts_db"
DB_FILE = "features_db.pkl"
ORB_FEATURES = 2000

# ==============================
# INIT
# ==============================
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

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
    _, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
# BUILD DATABASE
# ==============================
def build_database():
    database = {}

    for part in sorted(os.listdir(DB_PATH)):
        part_path = os.path.join(DB_PATH, part)
        if not os.path.isdir(part_path):
            continue

        print(f"[INFO] Processing part: {part}")
        database[part] = []

        for img_name in sorted(os.listdir(part_path)):
            img_path = os.path.join(part_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"  [WARN] Cannot read {img_name}")
                continue

            gray = preprocess(img)
            kp, des = orb.detectAndCompute(gray, None)
            shape = extract_shape_features(gray)

            if des is None or shape is None:
                print(f"  [WARN] Features not found in {img_name}")
                continue

            database[part].append({
                "des": des,
                "shape": shape
            })

            print(f"  [OK] {img_name}")

        if len(database[part]) == 0:
            print(f"  [ERROR] No valid samples for {part}")

    # Save database
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print("\n=== DATABASE BUILD COMPLETE ===")
    for part, samples in database.items():
        print(f"{part}: {len(samples)} samples")
    print("===============================\n")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    build_database()
