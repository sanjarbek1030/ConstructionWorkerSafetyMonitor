# 🦺 Construction Worker Safety Monitor

A real-time computer vision system that monitors construction site footage for **PPE (Personal Protective Equipment) compliance** and **danger zone intrusions** using **YOLOv8** and **OpenCV**.

The script detects workers in a video, checks whether they're wearing a helmet and safety vest, and flags anyone who steps into a predefined hazardous area — all in a single, beginner-friendly, procedural Python script.

---

## ✨ Features

- 🎥 **Video processing** — reads any `.mp4` input and writes a fully annotated output video at the same resolution and frame rate.
- 🧠 **YOLOv8 object detection** — detects `person`, `helmet`, and `safety-vest` (swap in your own custom-trained weights for real PPE detection).
- 🟥 **Danger zone overlay** — a translucent red polygon marks a hazardous area on the frame.
- 🚨 **Real-time violation alerts** — a flashing on-screen banner ("DANGER ZONE VIOLATION!") triggers the instant a worker's bounding box overlaps the danger zone.
- 🟢🔴 **PPE compliance boxes** — green boxes for workers with helmet + vest detected, red boxes for workers missing gear.
- 🖥️ **Live preview + file export** — watch the analysis happen in a window while it simultaneously saves to disk.
- 📝 **Fully commented, procedural code** — no classes, no frameworks to learn, just a clear top-to-bottom script anyone can follow and modify.

---

## 📸 Demo

| Input | Output |
|---|---|
| Raw construction site footage | Annotated video with PPE boxes, danger zone, and alerts |

*(Add your own before/after screenshots or a GIF here once you've run the script.)*

---

## 🛠️ Tech Stack

- [Python 3.9+](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/construction-safety-monitor.git
cd construction-safety-monitor

# (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy
```

---

## 🚀 Usage

1. Place your input video at:
   ```
   videos/input_video.mp4
   ```
2. Run the script:
   ```bash
   python safety_monitor.py
   ```
3. A live preview window opens while the script processes the video.
   Press **`q`** at any time to stop early.
4. The fully annotated video is saved to:
   ```
   videos/output_video.mp4
   ```

---

## ⚙️ Using a Custom PPE-Trained Model

Out of the box, this script uses `yolov8n.pt`, which is trained on the general-purpose **COCO** dataset. COCO includes a `person` class but **does not** include `helmet` or `safety-vest` classes — so by default, only person detection is meaningful.

For real PPE compliance detection, train (or download) a YOLOv8 model on a construction-safety dataset, such as one of the public PPE datasets on [Roboflow Universe](https://universe.roboflow.com/) (search "PPE detection yolov8"). Then simply update one line:

```python
MODEL_PATH = "path/to/your/best.pt"
```

The rest of the script automatically adapts — it scans the model's class names for anything containing `"person"`, `"helmet"`, or `"vest"`, so no other code changes are required.

---

## 🗺️ Customizing the Danger Zone

The danger zone is a hardcoded polygon defined near the top of the script:

```python
DANGER_ZONE_POLYGON_REFERENCE = np.array([
    [500, 300],
    [900, 300],
    [1000, 700],
    [400, 700],
], dtype=np.int32)
```

Edit these `(x, y)` points to match the hazardous area in your own footage (e.g. near heavy machinery, scaffolding, or an excavation pit). The polygon automatically scales to fit your video's resolution.

---

## 📁 Project Structure

```
construction-safety-monitor/
├── safety_monitor.py       # Main script — run this
├── videos/
│   ├── input_video.mp4     # Your source footage (add this yourself)
│   └── output_video.mp4    # Generated after running the script
└── README.md
```

---

## ⚠️ Limitations & Disclaimer

- This is a demonstration / educational project, **not a certified safety compliance tool**. Do not rely on it as a sole safety measure on an active job site.
- Detection accuracy depends entirely on the quality of the trained model and video conditions (lighting, occlusion, camera angle).
- PPE-overlap logic uses simple bounding-box intersection, not pose estimation, so it's an approximation of whether gear is actually being "worn."

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue if you'd like to suggest a feature (e.g. multi-camera support, gloves/goggles detection, Slack/email alerting, or a dashboard).

---

## 📄 License

This project is released under the [MIT License](LICENSE).
