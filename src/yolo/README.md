# YOLO Project Structure

This directory contains a small YOLO‑based vision system along with supporting scripts for building and managing a part database. The canonical layout and responsibilities of each file/module are outlined below.

> The structure described here is a guideline; you can extend it to suit your own experiment or deployment. Keep modules focused and document any new additions.

---

## 📁 Top‑level files

| File | Purpose |
|------|---------|
| `vision_system.py` | Main real‑time application. Loads a trained YOLO model (`best.pt`), performs inference on frames from a camera, annotates detections, and logs inspection results. This script is what you run during normal operation.
| `part_identifier.py` | Offline part‑recognition demo. Uses ORB features and shape descriptors to identify parts against a pre‑built database. Includes GUI loop and example logging, re‑used by the YOLO system for quality checks if integrated.
| `build_part_database.py` | Interactive capture utility. Lets the user press `c` to capture images from a camera, assign a part name, and automatically save features/shape vectors to a pickle file. Used to collect training samples.
| `build_database_images.py` | Batch database builder. Takes images already stored under `parts_db/<part_name>`, computes ORB/descriptors and shape features, and serialises them to the same `features_db.pkl` used by `part_identifier.py`. Helpful when you already have dataset images.
| `opencv.py` | Scratch/test script showing basic OpenCV operations (ROI selection, CamShift tracking). Included for reference; not required by the pipeline.

## 🗂️ Supporting directories

- `parts_db/` – directory containing subfolders for each recognized part. When using `build_part_database.py` or the batch builder, images are organised here:
  ```text
  parts_db/
    ├─ gear/
    │    ├─ 1.jpg
    │    ├─ 2.jpg
    │    └─ ...
    ├─ knob/
    └─ ...
  ```
  Each subfolder name becomes a key in the feature database.

## 📦 Generated artifacts

- `features_db.pkl` – pickled Python dictionary produced by either builder script. Keys are part names; values are lists of feature dictionaries containing ORB descriptors and shape vectors.
- `inspection_report.csv` – output from `vision_system.py` summarising detected parts and (eventually) quality decisions.

## 🛠️ How to extend

1. **Add a new module**: put additional utilities in a new `.py` file and document its role in this README.
2. **Model files**: the YOLO weights (`best.pt`) are expected in the same directory or a configurable path. You can keep additional models e.g. for defect detection.
3. **Data formats**: if you change how the database is stored, update the `DB_FILE` constants and add conversion helpers.
4. **Tests**: consider adding simple unit tests under a `tests/` folder matching the module names.

> 📝 _Tip_: keep configuration constants (`DB_PATH`, `CAMERA_INDEX`, etc.) at the top of each script so they are easy to override or refactor into a shared settings file.

---

## ✅ Summary
This structure separates concerns:
1. **Data acquisition** (`build_part_database.py`, `build_database_images.py`)
2. **Offline identification** (`part_identifier.py`)
3. **Live inspection using YOLO** (`vision_system.py`)
4. **Utility/demo** (`opencv.py`)

Following this layout makes it clear where to add new functionality and helps new contributors understand the project quickly.