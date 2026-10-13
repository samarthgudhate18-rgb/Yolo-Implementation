# YOLO Real-Time Object Detector

A **production-grade**, modular Python application for real-time object detection using
[Ultralytics YOLO](https://docs.ultralytics.com/) (YOLO11 / YOLOv8) and OpenCV,
with custom bounding-box visualisation, FPS overlay, class filtering, and multi-source support.

---

## Features

| Feature | Detail |
|---|---|
| **Multi-source input** | Webcam index, video file, static image |
| **Auto device selection** | CUDA GPU → Apple MPS → CPU fallback |
| **Streaming inference** | `model.predict(stream=True)` for memory-safe video |
| **Custom visualisation** | Semi-transparent fills, class-coloured borders, pill labels |
| **FPS overlay** | Exponential moving-average counter, top-left corner |
| **Info bar** | Model name, device, detection count — bottom bar |
| **Class filtering** | By name (`person`) or COCO ID (`0`), with fuzzy partial match |
| **Output saving** | Video (`.mp4`) or image (`.jpg` / `.png`) |
| **Clean exit** | Q / ESC keypress or window close button |

---

## Project Structure

```
YOLO/
├── detector.py        ← Main script (all logic lives here)
├── requirements.txt   ← Pinned dependencies
└── README.md
```

---

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

**CPU (all platforms)**
```bash
pip install -r requirements.txt
```

**NVIDIA GPU (CUDA 12.x)**
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (MPS)**
```bash
pip install -r requirements.txt
pip install torch torchvision   # ARM wheel includes MPS support
```

> **Note:** YOLO model weights (`yolo11n.pt`, `yolov8n.pt`, etc.) are downloaded
> automatically on first run from the Ultralytics hub.

---

## Usage

### Webcam (default — device index 0)

```bash
python detector.py --source 0
```

### Specific webcam index

```bash
python detector.py --source 1
```

### Video file

```bash
python detector.py --source path/to/video.mp4
```

### Static image

```bash
python detector.py --source path/to/image.jpg
```

---

## CLI Reference

```
usage: detector [-h] [--source SRC] [--model MODEL] [--conf CONF]
                [--iou IOU] [--classes CLS [CLS ...]] [--save OUT]
                [--device DEV]

Options:
  --source SRC          Input source: webcam index (0), video path, or image path.
                        (default: 0)

  --model  MODEL        YOLO weights: local path or Ultralytics name.
                        Pre-trained weights are auto-downloaded.
                        (default: yolo11n.pt)

  --conf   CONF         Confidence threshold in (0, 1].
                        (default: 0.45)

  --iou    IOU          NMS IoU overlap threshold in (0, 1].
                        (default: 0.50)

  --classes CLS …       Filter to specific classes by name or COCO ID.
                        Partial / fuzzy name matching is supported.

  --save   OUT          Output path to write the annotated result.
                        Video: output.mp4 | Image: result.jpg

  --device DEV          Force a specific device: cpu, cuda:0, mps.
                        (default: auto-detect)
```

---

## Examples

### Detect only people and laptops from webcam

```bash
python detector.py --source 0 --classes person laptop
```

### Filter by COCO class IDs

```bash
# COCO IDs: person=0, bicycle=1, car=2, cell phone=67, laptop=63
python detector.py --source 0 --classes 0 63 67
```

### Common target classes (COCO names)

```bash
python detector.py --source 0 --classes person "cell phone" "wine glass" laptop bottle cup
```

### High-precision run with GPU and video save

```bash
python detector.py \
  --source footage.mp4 \
  --model yolov8m.pt \
  --conf 0.60 \
  --iou 0.45 \
  --classes person \
  --device cuda:0 \
  --save detections.mp4
```

### Static image with custom threshold

```bash
python detector.py \
  --source photo.jpg \
  --conf 0.50 \
  --classes person laptop "cell phone" \
  --save annotated.jpg
```

### Force CPU inference (useful for debugging)

```bash
python detector.py --source 0 --device cpu
```

---

## Available YOLO Models

| Model | Size | Speed | mAP |
|---|---|---|---|
| `yolo11n.pt` | 2.6 MB | ⚡ Fastest | Good |
| `yolo11s.pt` | 9.4 MB | Fast | Better |
| `yolo11m.pt` | 20 MB | Moderate | Great |
| `yolo11l.pt` | 25 MB | Slow | Very good |
| `yolo11x.pt` | 56 MB | Slowest | Best |
| `yolov8n.pt` | 6.3 MB | ⚡ Fast | Good |
| `yolov8m.pt` | 25 MB | Moderate | Great |

---

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit / stop detection |
| Close window button | Quit |

---

## Common COCO Class IDs (80 classes)

| ID | Name | ID | Name | ID | Name |
|---|---|---|---|---|---|
| 0 | person | 24 | backpack | 47 | cup |
| 1 | bicycle | 25 | umbrella | 56 | chair |
| 2 | car | 26 | handbag | 62 | tv |
| 3 | motorcycle | 27 | tie | 63 | laptop |
| 14 | bird | 28 | suitcase | 64 | mouse |
| 15 | cat | 39 | bottle | 65 | remote |
| 16 | dog | 40 | wine glass | 66 | keyboard |
| 17 | horse | 41 | cup | 67 | cell phone |
| 63 | laptop | 42 | fork | 73 | book |
| 67 | cell phone | 43 | knife | 76 | scissors |

---

## Architecture Overview

```
detector.py
├── select_device()          — Auto-picks CUDA / MPS / CPU
├── _build_palette()         — Deterministic per-class colour generation
├── draw_detection()         — Semi-transparent box + pill label
├── draw_fps_overlay()       — EMA FPS counter (top-left)
├── draw_info_bar()          — Bottom info strip
├── FPSCounter               — Exponential moving-average FPS
├── YOLODetector             — Core class
│   ├── __init__()           — Model load + warm-up + class resolution
│   ├── _resolve_classes()   — Name/ID → int ID, fuzzy matching
│   ├── predict_stream()     — Generator: streaming video inference
│   ├── predict_image()      — Single image inference
│   └── _annotate()          — Applies draw_detection to all boxes
├── run_video_stream()       — Webcam/video loop + VideoWriter
├── run_image()              — Image display + save
├── build_arg_parser()       — argparse CLI definition
└── main()                   — Entry point
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: ultralytics` | Run `pip install -r requirements.txt` |
| Webcam not opening | Try `--source 1` or check device manager |
| CUDA out of memory | Use smaller model (`yolo11n.pt`) or `--device cpu` |
| Low FPS on CPU | Use `yolo11n.pt` (nano), reduce resolution, or use GPU |
| Class not found warning | Check spelling; use `--classes` with exact COCO name |
| Model download fails | Check internet or supply a local `.pt` file path |
