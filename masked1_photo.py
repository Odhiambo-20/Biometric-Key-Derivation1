"""
FACE EMBEDDING PIPELINE — Single Video
20 Frames | 20 Masked .jpg saved | 1 Final embedding

Phase 1  - Extract 20 evenly spaced frames from the video
Phase 2  - Detect and crop face from each frame (Haar cascade)
Phase 3  - Resize face crop to exactly 112 x 112
Phase 4  - Mask bottom 66% black  (rows 38..111 = 0)
           Keeps ONLY top 38 rows: forehead + upper eye area
           (nose, mouth, chin all hidden)
Phase 5  - Save each masked face as a .jpg for visual verification
Phase 6  - Extract 512-dim AdaFace embedding from each masked face
Phase 7  - Average all 20 embeddings into one vector
Phase 8  - L2 renormalize the averaged vector  (norm == 1.0)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── CONFIG ────────────────────────────────────────────────────────────────────

# ── Set this to whichever video you want to process ──────────────────────────
VIDEO_PATH = "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4"

WEIGHTS_PATH  = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
OUTPUT_ROOT   = "masked_frames"   # all 20 masked .jpg per video saved here
FRAMES_TO_USE = 20                # ALL 20 frames extracted, masked, and embedded
FACE_SIZE     = 112               # AdaFace input size

# Phase 4 mask:
#   VISIBLE  rows  0..37  = top 38 rows  (forehead + upper eye area only)
#   BLACK    rows 38..111 = 74 rows      (nose, mouth, chin, beard — all hidden)
#   MASK_FRACTION = 74/112 ≈ 0.66
MASK_FRACTION = 0.66


# ── MASK ──────────────────────────────────────────────────────────────────────

def apply_mask(image: np.ndarray) -> np.ndarray:
    """
    Black out the bottom MASK_FRACTION of the 112x112 image.
    rows  0..67  → visible  (forehead, eyebrows, eyes, nose)
    rows 68..111 → black    (mouth, chin, beard hidden)
    """
    img        = image.copy()
    h          = img.shape[0]
    black_rows = int(h * MASK_FRACTION)
    img[-black_rows:, :] = 0
    return img


# ── FACE DETECTOR ─────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(xml)
        if self.det.empty():
            raise RuntimeError("Haar cascade XML not found.")
        log.info("Face detector ready.")

    def detect(self, frame: np.ndarray, relax: bool = False) -> Optional[np.ndarray]:
        """
        Return tightly cropped face BGR array, or None.
        relax=True uses looser params to recover hard frames.
        """
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Try strict detection first; fall back to relaxed if relax=True
        configs = [
            dict(scaleFactor=1.05, minNeighbors=6, minSize=(80, 80)),
        ]
        if relax:
            configs.append(dict(scaleFactor=1.03, minNeighbors=3, minSize=(60, 60)))

        faces = []
        for cfg in configs:
            faces = self.det.detectMultiScale(gray, **cfg)
            if len(faces) > 0:
                break

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if w < 60 or h < 60:
            return None

        fh, fw = frame.shape[:2]
        x1 = max(0,  x - int(w * 0.10))
        y1 = max(0,  y - int(h * 0.05))
        x2 = min(fw, x + w + int(w * 0.10))
        y2 = min(fh, y + h + int(h * 0.02))

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None


# ── ADAFACE MODEL ─────────────────────────────────────────────────────────────

class AdaFaceModel:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded | Provider: {providers[0]}")

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        """112x112 BGR → 512-dim float32 embedding."""
        img = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        return emb.astype(np.float32)


# ── EXTRACT FRAMES ────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"  {Path(video_path).name} — {total} frames @ {fps:.1f} fps")

    positions = [int(round(i * (total - 1) / (num_frames - 1))) for i in range(num_frames)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((pos, frame))

    cap.release()
    log.info(f"  Extracted {len(frames)}/{num_frames} frames")
    return frames


# ── PROCESS ONE VIDEO ─────────────────────────────────────────────────────────

def process_video(
    video_path  : str,
    video_index : int,
    model       : AdaFaceModel,
    detector    : FaceDetector,
) -> Optional[np.ndarray]:

    sep  = "─" * 55
    name = Path(video_path).name
    print(f"\n{sep}\n  VIDEO {video_index}: {name}\n{sep}")

    # ── PHASE 1: Extract 20 evenly spaced frames ──────────────────────────────
    print("  [PHASE 1] Extracting 20 frames...")
    frames = extract_frames(video_path, FRAMES_TO_USE)
    if not frames:
        log.error("No frames extracted.")
        return None
    print(f"  → {len(frames)} frames extracted\n")

    # ── PHASE 2: Detect and crop face from each frame ─────────────────────────
    print("  [PHASE 2] Detecting faces...")
    crops          = []          # (frame_num, pos, crop, detected: bool)
    last_good_crop = None

    for frame_num, (pos, frame) in enumerate(frames, start=1):
        crop = detector.detect(frame, relax=False)

        if crop is None:
            crop = detector.detect(frame, relax=True)
            if crop is not None:
                log.info(f"  Frame {frame_num:>2} (pos {pos:>4}): detected (relaxed) {crop.shape[1]}x{crop.shape[0]}px")

        if crop is not None:
            last_good_crop = crop
            crops.append((frame_num, pos, crop, True))
            log.info(f"  Frame {frame_num:>2} (pos {pos:>4}): face detected ✓  {crop.shape[1]}x{crop.shape[0]}px")
        else:
            if last_good_crop is not None:
                crops.append((frame_num, pos, last_good_crop, False))
                log.warning(f"  Frame {frame_num:>2} (pos {pos:>4}): no face — reusing last good crop ↩")
            else:
                fallback = cv2.resize(frame, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
                crops.append((frame_num, pos, fallback, False))
                log.warning(f"  Frame {frame_num:>2} (pos {pos:>4}): no face — using raw frame (last resort) ⚠")

    detected_count = sum(1 for *_, ok in crops if ok)
    print(f"  → {detected_count}/{len(frames)} faces detected  |  {len(frames)-detected_count} fallback\n")

    if detected_count == 0:
        log.error(f"  No faces detected at all in {name} — aborting.")
        return None

    # ── PHASE 3: Resize every face crop to exactly 112x112 ───────────────────
    print("  [PHASE 3] Resizing all crops to 112x112...")
    resized_crops = []
    for frame_num, pos, crop, detected in crops:
        resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
        resized_crops.append((frame_num, pos, resized, detected))
        log.info(f"  Frame {frame_num:>2}: resized to {FACE_SIZE}x{FACE_SIZE}")
    print(f"  → All {len(resized_crops)} crops resized\n")

    # ── PHASE 4: Mask bottom 40% of each 112x112 face with pure black ─────────
    print(f"  [PHASE 4] Applying mask (bottom {int(MASK_FRACTION*100)}% = black)...")
    masked_crops = []
    for frame_num, pos, resized, detected in resized_crops:
        masked = apply_mask(resized)
        masked_crops.append((frame_num, pos, masked, detected))
        log.info(f"  Frame {frame_num:>2}: masked — rows {FACE_SIZE - int(FACE_SIZE*MASK_FRACTION)}..111 = black")
    print(f"  → All {len(masked_crops)} faces masked\n")

    # ── PHASE 5: Save ALL 20 masked faces as .jpg for visual verification ─────
    print("  [PHASE 5] Saving all masked faces to disk...")
    output_dir = Path(OUTPUT_ROOT) / f"video_{video_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_num, pos, masked, detected in masked_crops:
        save_path = output_dir / f"frame_{frame_num:02d}_pos{pos}{'_detected' if detected else '_fallback'}.jpg"
        cv2.imwrite(str(save_path), masked)
        log.info(f"  Frame {frame_num:>2}: saved → {save_path.name}")

    print(f"  → {len(masked_crops)} masked frames saved to: {output_dir}/\n")

    # ── PHASE 6: Extract 512-dim AdaFace embedding from each masked face ──────
    print("  [PHASE 6] Extracting AdaFace embeddings from all masked faces...")
    embeddings = []
    for frame_num, pos, masked, detected in masked_crops:
        emb = model.get_embedding(masked)      # AdaFace receives the masked 112x112 image
        embeddings.append(emb)
        log.info(f"  Frame {frame_num:>2}: embedding extracted  norm={np.linalg.norm(emb):.4f}  {'✓' if detected else '↩'}")

    assert len(embeddings) == FRAMES_TO_USE, \
        f"Expected {FRAMES_TO_USE} embeddings, got {len(embeddings)}"
    print(f"  → {len(embeddings)} embeddings extracted  (all {FRAMES_TO_USE} frames)\n")

    # ── PHASE 7: Average all embeddings into one vector ───────────────────────
    print("  [PHASE 7] Averaging all embeddings...")
    avg  = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    print(f"  → Averaged vector norm (pre-L2) : {np.linalg.norm(avg):.6f}\n")

    # ── PHASE 8: L2 renormalize — final norm must equal exactly 1.0 ──────────
    print("  [PHASE 8] L2 renormalizing...")
    norm = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Embedding norm near zero — cannot normalize.")
    final = (avg / norm).astype(np.float32)
    final_norm = float(np.linalg.norm(final))
    print(f"  → Final norm : {final_norm:.8f}  (must be 1.0)\n")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(sep)
    print(f"  SUMMARY — VIDEO {video_index}: {name}")
    print(sep)
    print(f"  Phase 1 — Frames extracted   : {len(frames)}")
    print(f"  Phase 2 — Faces detected     : {detected_count}  ({len(frames)-detected_count} fallback)")
    print(f"  Phase 3 — Crops resized      : {len(resized_crops)}")
    print(f"  Phase 4 — Faces masked       : {len(masked_crops)}")
    print(f"  Phase 5 — Frames saved to    : {output_dir}/")
    print(f"  Phase 6 — Embeddings         : {len(embeddings)}")
    print(f"  Phase 7 — Averaged norm      : {np.linalg.norm(avg):.6f}")
    print(f"  Phase 8 — Final norm (L2)    : {final_norm:.8f}  ✓")
    print(sep)
    print(f"\n  ALL 512 EMBEDDING VALUES:")
    print(sep)
    for i, val in enumerate(final):
        print(f"  Dim {i+1:>3}: {val:.8f}")
    print(sep)

    return final


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    sep          = "═" * 55
    black_rows   = int(FACE_SIZE * MASK_FRACTION)
    visible_rows = FACE_SIZE - black_rows

    print(sep)
    print("  ADAFACE PIPELINE  (single video)")
    print(f"  1 Video | {FRAMES_TO_USE} Frames | 1 Photo")
    print(sep)
    print(f"  Mask (bottom {int(MASK_FRACTION*100)}%) : {black_rows} rows black  (rows {visible_rows}..111)")
    print(f"  Visible (top {100-int(MASK_FRACTION*100)}%)  : {visible_rows} rows shown (rows 0..{visible_rows-1})")
    print(f"  Visible area covers : forehead + eyebrows + eyes + NOSE")
    print(f"  Black area covers   : mouth + chin + beard")
    print(f"  Photo saved to      : {OUTPUT_ROOT}/video_1_masked.jpg")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    emb = process_video(VIDEO_PATH, 1, model, detector)

    print(f"\n{sep}")
    print("  COMPLETE")
    print(sep)
    if emb is not None:
        print(f"  {Path(VIDEO_PATH).name}")
        print(f"  norm={np.linalg.norm(emb):.8f}  emb[:3]={[round(float(x),6) for x in emb[:3]]}")
    print(f"\n  Photo: {Path(OUTPUT_ROOT).resolve()}/video_1_masked.jpg")
    print(sep)

    return emb


if __name__ == "__main__":
    run()
