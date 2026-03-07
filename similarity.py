"""
AdaFace Cosine Similarity Pipeline
====================================
Pipeline per video:
  1. Extract top-20 sharpest frames (scan 90 candidates)
  2. Detect face bounding box (multi-scale Haar cascade)
  3. Detect eyes inside face region (eye cascade) for precise landmarks
     → fallback to geometry if eye detection fails
  4. Umeyama similarity transform → canonical 112×112 aligned face
  5. AdaFace IR-18 → raw 512-dim embedding
  6. L2 normalise → unit vector
  7. Average all unit vectors across frames
  8. L2 renormalise → final FP32 unit embedding
  9. Cosine similarity = dot product (all pairs)

People:
  V1, V2, V3, V4 = SAME PERSON → all pairs should be above threshold
  V5, V6, V7     = DIFFERENT PEOPLE → all pairs should be below threshold

Threshold: 0.75
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",                          # V1
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",             # V2
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",                    # V3
    "/home/victor/Documents/Desktop/Embeddings/Android no beard version 2.mp4",  # V4
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",             # V5
    "/home/victor/Documents/Desktop/Embeddings/IOS -Sha V6 .MOV",               # V6
    "/home/victor/Documents/Desktop/Embeddings/IOS - Rusl V7.mov",              # V7
]

WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)

VIDEO_NAMES = {
    "video_1": "V1 IOS-Beard",
    "video_2": "V2 IOS-NoBrd",
    "video_3": "V3 Andr-Beard",
    "video_4": "V4 Andr-NoBrd",
    "video_5": "V5 Android5",
    "video_6": "V6 Sha",
    "video_7": "V7 Rusl",
}

GENUINE_SET    = {"video_1", "video_2", "video_3", "video_4"}
FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3         # scan 60 candidates → keep top 20
FACE_SIZE      = 112
THRESHOLD      = 0.75

# AdaFace canonical 5-point reference landmarks (112×112)
# left-eye-centre, right-eye-centre, nose-tip, left-mouth, right-mouth
REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FACE ALIGNER
# ─────────────────────────────────────────────────────────────────────────────

class FaceAligner:
    """
    Face detection + eye-based landmark estimation + Umeyama transform.

    Uses OpenCV eye cascade inside the face ROI to get precise eye centres.
    Falls back to bounding-box geometry proportions if eye detection fails.
    No dlib — dlib's landmark ordering is incompatible with AdaFace reference.
    """

    def __init__(self):
        # Face cascade
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        if self.face_cascade.empty():
            raise RuntimeError("Face Haar cascade not found.")

        # Eye cascade (more precise than geometry for eye centres)
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.eye_ok = not self.eye_cascade.empty()
        if self.eye_ok:
            log.info("Eye cascade loaded — precise eye centres enabled.")
        else:
            log.info("Eye cascade not found — using geometry fallback.")

        # CLAHE for low-contrast / blurry frames
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        log.info("FaceAligner ready.")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _detect_face(self, gray: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Multi-level detection with progressively relaxed parameters."""
        for sf, mn, ms in [
            (1.05, 6, 60),
            (1.05, 3, 40),
            (1.10, 2, 30),
        ]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms)
            )
            if len(faces) > 0:
                return tuple(max(faces, key=lambda f: f[2] * f[3]))
        return None

    def _detect_eyes(
        self, gray: np.ndarray, fx: int, fy: int, fw: int, fh: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect two eyes inside the face ROI.
        Returns (left_eye_centre, right_eye_centre) in full-image coordinates,
        or None if fewer than 2 eyes found.
        """
        if not self.eye_ok:
            return None

        # Only search in the top 60% of the face (eyes are never in lower half)
        eye_roi_h = int(fh * 0.60)
        roi       = gray[fy : fy + eye_roi_h, fx : fx + fw]

        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20)
        )
        if len(eyes) < 2:
            # Relax detection for blurry frames
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.10, minNeighbors=2, minSize=(15, 15)
            )
        if len(eyes) < 2:
            return None

        # Keep the two largest eyes
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        centres = []
        for (ex, ey, ew, eh) in eyes:
            cx = fx + ex + ew // 2
            cy = fy + ey + eh // 2
            centres.append(np.array([cx, cy], dtype=np.float32))

        # Sort left-to-right
        centres.sort(key=lambda p: p[0])
        return centres[0], centres[1]   # left eye, right eye

    @staticmethod
    def _landmarks_from_box(
        x: int, y: int, w: int, h: int,
        left_eye: Optional[np.ndarray] = None,
        right_eye: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build 5-point landmarks.
        Uses detected eye centres if available, else geometry estimates.
        Nose and mouth are always geometry-based (cascade not available).
        """
        if left_eye is not None and right_eye is not None:
            le = left_eye
            re = right_eye
        else:
            le = np.array([x + 0.30 * w, y + 0.36 * h], dtype=np.float32)
            re = np.array([x + 0.70 * w, y + 0.36 * h], dtype=np.float32)

        nose  = np.array([x + 0.50 * w,  y + 0.57 * h], dtype=np.float32)
        lmouth = np.array([x + 0.35 * w, y + 0.76 * h], dtype=np.float32)
        rmouth = np.array([x + 0.65 * w, y + 0.76 * h], dtype=np.float32)

        return np.stack([le, re, nose, lmouth, rmouth]).astype(np.float32)

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        """Umeyama least-squares similarity transform → 2×3 affine matrix."""
        n    = src.shape[0]
        mu_s = src.mean(0);  mu_d = dst.mean(0)
        sc   = src - mu_s;   dc   = dst - mu_d
        vs   = (sc ** 2).sum() / n
        if vs < 1e-10:
            return None
        cov = (dc.T @ sc) / n
        try:
            U, S, Vt = np.linalg.svd(cov)
        except np.linalg.LinAlgError:
            return None
        d = np.ones(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            d[-1] = -1
        R = U @ np.diag(d) @ Vt
        c = (S * d).sum() / vs
        t = mu_d - c * R @ mu_s
        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = c * R
        M[:, 2]  = t
        return M

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face → detect eyes inside face → build 5-point landmarks
        → Umeyama warp → 112×112 BGR aligned face.
        Returns None if no face detected.
        """
        gray = self._preprocess(frame)
        det  = self._detect_face(gray)
        if det is None:
            return None
        x, y, w, h = det

        # Try eye detection inside face region
        eye_result = self._detect_eyes(gray, x, y, w, h)
        left_eye   = eye_result[0] if eye_result else None
        right_eye  = eye_result[1] if eye_result else None

        src_pts = self._landmarks_from_box(x, y, w, h, left_eye, right_eye)
        M       = self._umeyama(src_pts, REFERENCE_PTS)

        if M is None:
            fh, fw = frame.shape[:2]
            x1 = max(0, x - int(w * 0.10))
            y1 = max(0, y - int(h * 0.05))
            x2 = min(fw, x + w + int(w * 0.10))
            y2 = min(fh, y + h + int(h * 0.02))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, (FACE_SIZE, FACE_SIZE),
                              interpolation=cv2.INTER_LANCZOS4)

        return cv2.warpAffine(
            frame, M, (FACE_SIZE, FACE_SIZE),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ADAFACE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AdaFaceModel:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"AdaFace IR-18 | {providers[0]}")

    def raw_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR 112×112 → raw 512-dim float32 (not yet normalised)."""
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> List[Tuple[float, np.ndarray]]:
    """
    Scan CANDIDATE_MULT × FRAMES_TO_USE positions.
    Return top FRAMES_TO_USE by Laplacian sharpness in temporal order.
    Returns list of (sharpness_score, frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan    = FRAMES_TO_USE * CANDIDATE_MULT
    positions = [int(round(i * (total - 1) / max(n_scan - 1, 1)))
                 for i in range(n_scan)]

    candidates = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        candidates.append((score, pos, frame))
    cap.release()

    if not candidates:
        raise RuntimeError(f"No frames in {video_path}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:FRAMES_TO_USE]
    top.sort(key=lambda x: x[1])
    return [(score, frame) for score, _, frame in top]


# ─────────────────────────────────────────────────────────────────────────────
#  EMBED ONE VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def embed_video(
    video_path : str,
    model      : AdaFaceModel,
    aligner    : FaceAligner,
) -> Optional[np.ndarray]:
    """
    Full pipeline for one video → single FP32 unit embedding.

    For each frame:
      → align face (eye cascade + Umeyama → 112×112)
      → raw AdaFace embedding
      → L2 normalise → unit vector

    Average all unit vectors → L2 renormalise → final unit embedding.
    """
    frames     = extract_frames(video_path)
    unit_vecs  = []
    face_count = 0

    for _, frame in frames:
        aligned = aligner.align(frame)
        if aligned is None:
            continue
        face_count += 1
        raw  = model.raw_embedding(aligned)
        norm = np.linalg.norm(raw)
        if norm < 1e-10:
            continue
        unit_vecs.append((raw / norm).astype(np.float32))

    if not unit_vecs:
        log.error(f"No faces: {Path(video_path).name}")
        return None

    avg      = np.mean(np.stack(unit_vecs, axis=0), axis=0).astype(np.float32)
    avg_norm = np.linalg.norm(avg)
    if avg_norm < 1e-10:
        log.error(f"Near-zero average: {Path(video_path).name}")
        return None

    final = (avg / avg_norm).astype(np.float32)
    log.info(
        f"  {Path(video_path).name:<44}"
        f"  faces={face_count:>2}/{len(frames)}"
        f"  norm={np.linalg.norm(final):.6f}"
    )
    return final


# ─────────────────────────────────────────────────────────────────────────────
#  COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model   = AdaFaceModel(WEIGHTS_PATH)
    aligner = FaceAligner()

    # ── Embed all 7 videos ───────────────────────────────────────────────────
    print("\nExtracting embeddings...")
    print("─" * 65)

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        key    = f"video_{idx}"
        result = embed_video(vp, model, aligner)
        if result is not None:
            embeddings[key] = result

    # ── All-vs-all cosine similarity ─────────────────────────────────────────
    keys = [f"video_{i}" for i in range(1, 8) if f"video_{i}" in embeddings]

    print("\n")
    print("=" * 55)
    print("  COSINE SIMILARITY — ALL PAIRS")
    print(f"  Threshold = {THRESHOLD}")
    print("=" * 55)
    print()
    print(f"  {'Pair':<35}  {'Similarity':>10}")
    print(f"  {'─'*35}  {'─'*10}")

    pair_results = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            sim          = cosine_sim(embeddings[ka], embeddings[kb])
            na           = VIDEO_NAMES.get(ka, ka)
            nb           = VIDEO_NAMES.get(kb, kb)
            both_genuine = ka in GENUINE_SET and kb in GENUINE_SET
            verdict      = "PASS" if sim >= THRESHOLD else "FAIL"
            pair_results.append((ka, kb, na, nb, sim, both_genuine, verdict))
            print(f"  {na + ' vs ' + nb:<35}  {sim:>10.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    genuine_sims  = [sim for *_, sim, bg, v in pair_results if bg]
    impostor_sims = [sim for *_, sim, bg, v in pair_results if not bg]

    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)

    if genuine_sims:
        print(f"  Genuine  pairs (V1–V4)"
              f"  min={min(genuine_sims):.4f}"
              f"  max={max(genuine_sims):.4f}"
              f"  mean={np.mean(genuine_sims):.4f}")

    if impostor_sims:
        print(f"  Impostor pairs (V5–V7)"
              f"  min={min(impostor_sims):.4f}"
              f"  max={max(impostor_sims):.4f}"
              f"  mean={np.mean(impostor_sims):.4f}")

    if genuine_sims and impostor_sims:
        gap = min(genuine_sims) - max(impostor_sims)
        print(f"  Separation gap          {gap:+.4f}"
              f"  {'CLEAN ✓' if gap > 0 else 'OVERLAP ✗'}")

    print()
    wrong = [
        (na, nb, sim, bg, v)
        for ka, kb, na, nb, sim, bg, v in pair_results
        if (bg and v == "FAIL") or (not bg and v == "PASS")
    ]

    if not wrong:
        print(f"  ALL PAIRS CORRECT ✓")
        print(f"  Genuine  → all above {THRESHOLD}")
        print(f"  Impostor → all below {THRESHOLD}")
    else:
        print(f"  {len(wrong)} PAIR(S) WRONG ✗")
        for na, nb, sim, bg, v in wrong:
            expected = "PASS" if bg else "FAIL"
            print(f"    {na} vs {nb}  sim={sim:.4f}"
                  f"  got={v}  expected={expected}")

    print("=" * 55)


if __name__ == "__main__":
    main()
