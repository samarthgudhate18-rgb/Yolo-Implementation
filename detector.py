"""
detector.py — Production-grade real-time object detection with Ultralytics YOLO + OpenCV.

Author  : Senior CV/Python Engineer
Version : 1.0.0
Python  : 3.10+

Usage examples
--------------
  # Webcam (index 0)
  python detector.py --source 0

  # Video file
  python detector.py --source path/to/video.mp4 --conf 0.5 --save output.mp4

  # Static image
  python detector.py --source path/to/image.jpg --save result.jpg

  # Custom class filter + custom model
  python detector.py --source 0 --model yolov8n.pt --classes person laptop "cell phone"

  # High-precision run on GPU
  python detector.py --source 0 --conf 0.6 --iou 0.45 --classes person bottle
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("YOLODetector")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_TITLE: str = "YOLO Real-Time Detector  |  [Q / ESC] to quit"

# Colour palette: deterministic HSV → BGR so each class gets a unique hue.
_PALETTE_N: int = 80  # enough for COCO (80 classes)


def _build_palette(n: int) -> list[tuple[int, int, int]]:
    """
    Generate *n* perceptually distinct BGR colours using the HSV colour space.
    The palette is deterministic (seed is fixed) so class colours are stable
    across frames and runs.

    Args:
        n: Number of distinct colours to generate.

    Returns:
        List of BGR colour tuples.
    """
    rng = np.random.default_rng(seed=42)
    hues = np.linspace(0, 179, n, dtype=np.uint8)  # OpenCV H ∈ [0, 179]
    rng.shuffle(hues)
    palette: list[tuple[int, int, int]] = []
    for h in hues:
        hsv_pixel = np.array([[[h, 200, 220]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])))
    return palette


_PALETTE: list[tuple[int, int, int]] = _build_palette(_PALETTE_N)


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def select_device() -> str:
    """
    Dynamically detect the best available compute device.

    Priority order: CUDA GPU → Apple Silicon MPS → CPU.

    Returns:
        Device string understood by PyTorch / Ultralytics (e.g. 'cuda:0', 'mps', 'cpu').
    """
    try:
        if torch.cuda.is_available():
            device = "cuda:0"
            name = torch.cuda.get_device_name(0)
            logger.info("✅  CUDA GPU detected — using '%s' (%s)", device, name)
            return device

        # Apple Silicon MPS (macOS 12.3+, PyTorch ≥ 1.12)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("✅  Apple MPS detected — using 'mps'")
            return "mps"

    except Exception as exc:  # pragma: no cover
        logger.warning("Device probe failed (%s). Falling back to CPU.", exc)

    logger.info("⚠️  No GPU found — using CPU (inference will be slower)")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _colour_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for the given class ID."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detection(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    class_id: int,
    class_name: str,
    confidence: float,
    box_alpha: float = 0.25,
    text_scale: float = 0.60,
    text_thickness: int = 1,
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Draw a single detection onto *frame* with:
      - A semi-transparent filled rectangle (box fill).
      - A solid border rectangle in the class colour.
      - A pill-shaped label banner with class name + confidence.

    Args:
        frame         : BGR image array (modified in-place and returned).
        xyxy          : Bounding-box coordinates (x1, y1, x2, y2).
        class_id      : Integer class index (used to pick colour).
        class_name    : Human-readable class label.
        confidence    : Detection confidence in [0, 1].
        box_alpha     : Opacity of the filled box overlay (0 = transparent, 1 = opaque).
        text_scale    : Font scale for the label text.
        text_thickness: Thickness of the label text strokes.
        box_thickness : Thickness of the bounding-box border.

    Returns:
        The annotated frame (same array as input).
    """
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    colour = _colour_for_class(class_id)

    # ── Semi-transparent box fill ─────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, thickness=-1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)

    # ── Solid border ─────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness=box_thickness)

    # ── Label banner ─────────────────────────────────────────────────────────
    label = f"{class_name}  {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
    pad = 4

    # Clamp banner so it never goes above the frame
    banner_y1 = max(y1 - th - 2 * pad, 0)
    banner_y2 = banner_y1 + th + 2 * pad

    # Filled banner background
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 2 * pad, banner_y2), colour, thickness=-1)

    # Dark or light text depending on colour brightness
    brightness = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
    text_colour = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    cv2.putText(
        frame,
        label,
        (x1 + pad, banner_y2 - pad - baseline // 2),
        font,
        text_scale,
        text_colour,
        text_thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def draw_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Render a real-time FPS counter in the top-left corner of *frame*.

    Args:
        frame: BGR image array (modified in-place and returned).
        fps  : Current frames-per-second value.

    Returns:
        The annotated frame.
    """
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 0.65, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    margin = 10

    # Background pill
    cv2.rectangle(
        frame,
        (margin, margin),
        (margin + tw + 2 * pad, margin + th + 2 * pad),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (margin, margin),
        (margin + tw + 2 * pad, margin + th + 2 * pad),
        (0, 200, 100),
        thickness=1,
    )
    # Text
    cv2.putText(
        frame,
        label,
        (margin + pad, margin + th + pad - baseline // 2),
        font,
        scale,
        (0, 220, 120),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def draw_info_bar(
    frame: np.ndarray,
    model_name: str,
    device: str,
    n_detections: int,
) -> np.ndarray:
    """
    Draw a slim info bar at the bottom of *frame* with model name,
    device, and detection count.

    Args:
        frame       : BGR image array (modified in-place and returned).
        model_name  : Name of the YOLO model in use.
        device      : Compute device string.
        n_detections: Number of objects detected in the current frame.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]
    bar_h = 26
    bar_y = h - bar_h

    # Semi-transparent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (10, 10, 10), thickness=-1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    info = f"  Model: {model_name}   Device: {device}   Detections: {n_detections}"
    cv2.putText(
        frame,
        info,
        (4, h - 7),
        cv2.FONT_HERSHEY_PLAIN,
        0.95,
        (180, 180, 180),
        1,
        lineType=cv2.LINE_AA,
    )
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Core detector class
# ──────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Encapsulates Ultralytics YOLO model loading, inference, and custom
    OpenCV visualisation.  Supports webcam, video files, and static images.

    Attributes:
        model      : Loaded Ultralytics YOLO model.
        device     : Compute device string.
        conf       : Confidence threshold.
        iou        : NMS IoU threshold.
        class_ids  : Optional list of integer class IDs to keep (None = all).
        model_name : Basename of the model file used.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.45,
        iou: float = 0.50,
        class_filter: Optional[list[str | int]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise the detector.

        Args:
            model_path   : Path (or name) of the YOLO weights file.
                           Ultralytics will download pre-trained weights automatically
                           if the file is not found locally.
            conf         : Minimum confidence threshold for detections.
            iou          : NMS IoU overlap threshold.
            class_filter : List of class names (str) or class IDs (int) to keep.
                           Pass None to detect all classes.
            device       : Compute device override.  If None, auto-selected.
        """
        self.conf = conf
        self.iou = iou
        self.model_name = Path(model_path).name
        self.device = device or select_device()

        logger.info("Loading model '%s' …", model_path)
        try:
            self.model: YOLO = YOLO(model_path)
            # Warm-up pass to initialise CUDA kernels / MPS graph
            logger.info("Running model warm-up …")
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", model_path, exc)
            sys.exit(1)

        # Resolve class filter to integer IDs using the model's names dict
        self.class_ids: Optional[list[int]] = self._resolve_classes(class_filter)

        if self.class_ids is not None:
            names = [self.model.names[i] for i in self.class_ids]
            logger.info("Filtering to %d class(es): %s", len(names), names)
        else:
            logger.info("No class filter — detecting all %d classes.", len(self.model.names))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_classes(
        self, class_filter: Optional[list[str | int]]
    ) -> Optional[list[int]]:
        """
        Convert a mixed list of class names / IDs into a sorted list of integer IDs
        that exist in the model's class dictionary.

        Args:
            class_filter: Raw filter values from the CLI.

        Returns:
            Sorted list of valid integer class IDs, or None if no filter provided.

        Raises:
            SystemExit: If none of the requested classes exist in the model.
        """
        if not class_filter:
            return None

        names_lower = {v.lower(): k for k, v in self.model.names.items()}
        resolved: list[int] = []
        for item in class_filter:
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                cid = int(item)
                if cid in self.model.names:
                    resolved.append(cid)
                else:
                    logger.warning("Class ID %d not found in model — skipping.", cid)
            else:
                key = str(item).lower()
                if key in names_lower:
                    resolved.append(names_lower[key])
                else:
                    # Fuzzy partial match
                    matches = [v for k, v in names_lower.items() if key in k]
                    if matches:
                        resolved.extend(matches)
                        logger.info(
                            "Class '%s' fuzzy-matched → %s",
                            item,
                            [self.model.names[m] for m in matches],
                        )
                    else:
                        logger.warning("Class '%s' not found in model — skipping.", item)

        if not resolved:
            logger.error(
                "None of the requested classes %s were found in model '%s'.",
                class_filter,
                self.model_name,
            )
            sys.exit(1)

        return sorted(set(resolved))

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_stream(
        self, source: str | int
    ) -> Generator[tuple[np.ndarray, Results], None, None]:
        """
        Run streaming inference over a video source.

        Uses ``model.predict(stream=True)`` so that frames are processed
        lazily and GPU memory is not exhausted during long captures.

        Args:
            source: Webcam index (int), video file path (str), or image path (str).

        Yields:
            Tuple of (annotated_frame, raw_Results) for each input frame.
        """
        for result in self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_ids,
            stream=True,
            verbose=False,
        ):
            frame = result.orig_img.copy()
            frame = self._annotate(frame, result)
            yield frame, result

    def predict_image(self, source: str) -> np.ndarray:
        """
        Run inference on a single static image (non-streaming).

        Args:
            source: Path to the input image file.

        Returns:
            Annotated BGR image array.
        """
        results = self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_ids,
            verbose=False,
        )
        frame = results[0].orig_img.copy()
        return self._annotate(frame, results[0])

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, result: Results) -> np.ndarray:
        """
        Apply custom OpenCV bounding-box annotations to *frame*.

        Iterates over all detections in *result* and calls :func:`draw_detection`
        for each one.

        Args:
            frame : BGR image to annotate (modified in-place).
            result: Ultralytics ``Results`` object for this frame.

        Returns:
            Annotated frame.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return frame

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names.get(class_id, str(class_id))
            confidence = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            draw_detection(
                frame=frame,
                xyxy=xyxy,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
            )

        return frame


# ──────────────────────────────────────────────────────────────────────────────
# Source routing — webcam / video / image
# ──────────────────────────────────────────────────────────────────────────────

def _is_image(source: str) -> bool:
    """Return True if *source* looks like a static image path."""
    return Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".avif",
    }


def run_video_stream(
    detector: YOLODetector,
    source: str | int,
    save_path: Optional[str] = None,
) -> None:
    """
    Run real-time detection on a webcam or video file, rendering results
    in an interactive OpenCV window.

    Args:
        detector  : Initialised :class:`YOLODetector` instance.
        source    : Webcam index or path to a video file.
        save_path : Optional output video path.  If None, no file is written.
    """
    logger.info("Opening video source: %s", source)

    # Open capture just to query frame dimensions for the writer
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer: Optional[cv2.VideoWriter] = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, src_fps, (frame_w, frame_h))
        logger.info("Saving output to: %s", save_path)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # FPS tracking
    fps_counter = FPSCounter(smoothing=0.9)

    try:
        for annotated_frame, result in detector.predict_stream(source):
            fps = fps_counter.tick()
            n_det = len(result.boxes) if result.boxes is not None else 0

            draw_fps_overlay(annotated_frame, fps)
            draw_info_bar(annotated_frame, detector.model_name, detector.device, n_det)

            cv2.imshow(WINDOW_TITLE, annotated_frame)

            if writer is not None:
                writer.write(annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                logger.info("Exit key pressed — stopping.")
                break

            # Handle window close button (WM_DELETE)
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed — stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        if writer is not None:
            writer.release()
            logger.info("Video saved: %s", save_path)
        cv2.destroyAllWindows()
        logger.info("Done.")


def run_image(
    detector: YOLODetector,
    source: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Run detection on a single static image, display result, and optionally save.

    Args:
        detector  : Initialised :class:`YOLODetector` instance.
        source    : Path to the input image file.
        save_path : Optional output image path.
    """
    logger.info("Processing image: %s", source)
    annotated = detector.predict_image(source)
    n_det = 0  # already counted inside predict_image; cosmetic here

    draw_info_bar(annotated, detector.model_name, detector.device, n_det)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_TITLE, annotated)

    if save_path:
        cv2.imwrite(save_path, annotated)
        logger.info("Annotated image saved: %s", save_path)

    logger.info("Press any key in the image window to exit …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# FPS counter
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Exponential moving-average FPS counter.

    Args:
        smoothing: EMA alpha — higher values give slower, smoother readout.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        self._alpha = smoothing
        self._fps: float = 0.0
        self._last: float = time.perf_counter()

    def tick(self) -> float:
        """
        Record a new frame and return the smoothed FPS.

        Returns:
            Current smoothed FPS value.
        """
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps = self._alpha * self._fps + (1 - self._alpha) * instant_fps
        return self._fps


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="detector",
        description=(
            "Real-time object detection with Ultralytics YOLO + OpenCV.\n"
            "Supports webcam, video file, and static image sources."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python detector.py --source 0
  python detector.py --source video.mp4 --save output.mp4
  python detector.py --source image.jpg --save result.jpg --conf 0.5
  python detector.py --source 0 --classes person laptop "cell phone"
  python detector.py --source 0 --model yolov8n.pt --conf 0.6 --iou 0.45
""",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        metavar="SRC",
        help=(
            "Input source.  Options:\n"
            "  0          → default webcam\n"
            "  <N>        → webcam index N\n"
            "  <path>     → video file (.mp4, .avi, …)\n"
            "  <path>     → image file (.jpg, .png, …)\n"
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        metavar="MODEL",
        help=(
            "YOLO model weights.  Local path or Ultralytics hub name.\n"
            "Pre-trained weights are auto-downloaded if not found.\n"
            "(default: yolo11n.pt)"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        metavar="CONF",
        help="Confidence threshold for detections, in [0, 1]. (default: 0.45)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        metavar="IOU",
        help="NMS IoU overlap threshold, in [0, 1]. (default: 0.50)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLS",
        help=(
            "Filter to specific class names or IDs.  Examples:\n"
            "  --classes person laptop\n"
            "  --classes 0 67         (COCO IDs: person=0, cell phone=67)\n"
            "  --classes person 'cell phone' 'wine glass'\n"
            "(default: detect all classes)"
        ),
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="OUT",
        help=(
            "Output path to save the annotated result.\n"
            "  For video/webcam: e.g. output.mp4\n"
            "  For image      : e.g. result.jpg\n"
            "(default: no save)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEV",
        help=(
            "Compute device override.  Examples: cpu, cuda:0, mps.\n"
            "(default: auto-detect)"
        ),
    )
    return parser


def main() -> None:
    """Entry point: parse CLI args, build detector, dispatch to source handler."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate thresholds
    if not (0.0 < args.conf <= 1.0):
        parser.error(f"--conf must be in (0, 1], got {args.conf}")
    if not (0.0 < args.iou <= 1.0):
        parser.error(f"--iou must be in (0, 1], got {args.iou}")

    detector = YOLODetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        class_filter=args.classes,
        device=args.device,
    )

    # Dispatch
    source = args.source
    if _is_image(source):
        run_image(detector, source, save_path=args.save)
    else:
        # Webcam index or video file
        run_video_stream(detector, source, save_path=args.save)


if __name__ == "__main__":
    main()
