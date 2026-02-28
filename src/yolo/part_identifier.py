import cv2
import numpy as np
import os
import pickle

# ==============================
# CONFIG
# ==============================
DB_PATH = "parts_db"
DB_FILE = "features_db.pkl"
CAMERA_INDEX = 0
ORB_FEATURES = 2000

# ==============================
# INITIALIZE
# ==============================
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    shape_vec = np.hstack([area, perimeter, aspect_ratio, hu])

    return shape_vec, (x, y, w, h)

# ==============================
# BUILD DATABASE
# ==============================
def build_database():
    database = {}

    for part in os.listdir(DB_PATH):
        part_path = os.path.join(DB_PATH, part)
        if not os.path.isdir(part_path):
            continue

        database[part] = []

        for img_name in os.listdir(part_path):
            img_path = os.path.join(part_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = preprocess(img)
            kp, des = orb.detectAndCompute(gray, None)
            shape_vec, _ = extract_shape_features(gray)

            if des is not None and shape_vec is not None:
                database[part].append({
                    "des": des,
                    "shape": shape_vec
                })

    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print("[INFO] Database built and saved.")

# ==============================
# MATCHING
# ==============================
def orb_score(d1, d2):
    matches = bf.match(d1, d2)
    if len(matches) < 10:
        return 1e9
    matches = sorted(matches, key=lambda x: x.distance)
    return np.mean([m.distance for m in matches[:30]])

def shape_score(s1, s2):
    return np.linalg.norm(s1 - s2)

# ==============================
# IDENTIFICATION
# ==============================
def identify(gray):
    kp, des = orb.detectAndCompute(gray, None)
    shape_vec, bbox = extract_shape_features(gray)

    if des is None or shape_vec is None:
        return None, 0, None

    best_part = None
    best_score = float("inf")

    for part, samples in database.items():
        for sample in samples:
            o_score = orb_score(des, sample["des"])
            s_score = shape_score(shape_vec, sample["shape"])
            final_score = 0.7 * o_score + 0.3 * s_score

            if final_score < best_score:
                best_score = final_score
                best_part = part

    confidence = max(0, 100 - best_score / 10)
    return best_part, confidence, bbox

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    # Build DB once if not exists
    if not os.path.exists(DB_FILE):
        build_database()

    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.read

        frame = cv2.resize(frame, (640, 480))
        gray = preprocess(frame)

        part, conf, bbox = identify(gray)

        if part and bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{part} ({conf:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            print(part)

        cv2.imshow("Part Identifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
