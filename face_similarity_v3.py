"""
AdaFace Fine-Tuning Pipeline — Sub-center + Makeup Augmentation + Hard Triplet Mining
=======================================================================================
Builds on the Bilateral Filter inference pipeline (Run 1 best results).

Three improvements combined:
  1. Sub-center AdaFace loss (K=3 prototypes per identity)
  2. Makeup augmentation (synthetic foundation / eyeshadow / lipstick)
  3. Hard triplet mining (online, semi-hard + hard strategy)

Data layout expected:
  VIDEO_DIR/
    V8 instagram Make up.mp4
    V8 instagram No make up.mp4
    ...

Frame extraction (100 frames per video) -> alignment -> bilateral filter ->
colour normalisation -> augmentation -> fine-tune IR-18 head.

Outputs:
  finetuned_adaface_ir18.pth   — fine-tuned weights (PyTorch)
  finetune_log.csv             — per-epoch loss / within-id similarity
  eval_matrix.txt              — 12x12 cosine similarity after fine-tuning
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
#  CONFIG
# =============================================================================

VIDEO_DIR    = Path("/home/victor/Documents/Desktop/Embeddings/drive-download-20260310T203758Z-1-001")
WEIGHTS_PATH = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir18_torch.pth"
OUTPUT_DIR   = Path("/home/victor/Documents/Desktop/Adaface/finetuned")

VIDEOS = [
    ("V8",  "makeup",    "V8 instagram Make up.mp4"),
    ("V8",  "no_makeup", "V8 instagram No make up.mp4"),
    ("V9",  "makeup",    "V9 Instagram Make up.mp4"),
    ("V9",  "no_makeup", "V9 Instagram No make up.mp4"),
    ("V10", "makeup",    "V10 Instagram Make up.mp4"),
    ("V10", "no_makeup", "V10 Instagram No make up.mp4"),
    ("V11", "makeup",    "V11 instagram Make up.mp4"),
    ("V11", "no_makeup", "V11 instagram No make up.mp4"),
    ("V12", "makeup",    "V12 instagram Make up.mp4"),
    ("V12", "no_makeup", "V12 instagram No make up.mp4"),
    ("V13", "makeup",    "V13 instagram Make up.mp4"),
    ("V13", "no_makeup", "V13 instagram No make up.mp4"),
]

PERSONS = ["V8", "V9", "V10", "V11", "V12", "V13"]
NUM_CLASSES = len(PERSONS)

# Frame extraction
FRAMES_TO_USE    = 100     # increased from 20 for training
CANDIDATE_MULT   = 3
FACE_SIZE        = 112
MIN_FACE_PX      = 40

# Bilateral filter
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Colour normalisation
TARGET_LUMINANCE = 130.0
LAB_CLAHE        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
NEUTRAL_A        = 128.0
NEUTRAL_B        = 128.0

# Sub-center
K_SUBCENTERS = 3

# Training
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 5e-4
NUM_EPOCHS      = 30
BATCH_SIZE      = 16
TRIPLET_WEIGHT  = 0.1    # weight of triplet loss relative to AdaFace loss
TRIPLET_MARGIN  = 0.3
EYE_DIST_TOL    = 0.20
THRESHOLD       = 0.75

# Augmentation styles: (lip_b, lip_g, lip_r, eye_b, eye_g, eye_r, foundation_shift)
# BGR order for OpenCV
MAKEUP_STYLES = [
    # Classic red lip, subtle eye
    {"lip": (30,  20, 180), "eye": (40,  40, 100), "foundation": 5},
    # Nude lip, smoky eye
    {"lip": (90,  80, 130), "eye": (20,  20,  20), "foundation": 8},
    # Berry lip, blue eye
    {"lip": (80,  20, 130), "eye": (120, 40,  30), "foundation": 6},
    # Coral lip, bronze eye
    {"lip": (40, 120, 200), "eye": (20,  80, 140), "foundation": 10},
]

# 5-point reference landmarks for 112x112 crop
REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# =============================================================================
#  STEP 1b — BILATERAL FILTER
# =============================================================================

def bilateral_smooth(face_bgr: np.ndarray) -> np.ndarray:
    """
    Edge-preserving bilateral filter on the aligned 112x112 crop.
    Smooths makeup texture (foundation, powder) while preserving structural
    edges (nose bridge, jaw, cheekbones).
    Applied BEFORE colour normalisation.
    """
    return cv2.bilateralFilter(
        face_bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )


# =============================================================================
#  STEP 2 — COLOUR NORMALISATION
# =============================================================================

def normalise_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Colour normalisation on the bilateral-smoothed aligned crop:
      a. Gamma correction -> target luminance 130
      b. LAB CLAHE on L channel
      c. A/B channel mean-shift to neutral (removes foundation colour cast)
    """
    h, w   = face_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    my, mx = h // 5, w // 5
    grey   = cv2.cvtColor(face_bgr[cy-my:cy+my, cx-mx:cx+mx], cv2.COLOR_BGR2GRAY)
    mean_l = float(grey.mean())
    if mean_l > 5.0:
        gamma = float(np.clip(
            np.log(TARGET_LUMINANCE / 255.0) / np.log(mean_l / 255.0), 0.3, 3.0))
        lut = np.array(
            [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)],
            dtype=np.uint8)
        face_bgr = lut[face_bgr]
    lab     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq    = LAB_CLAHE.apply(l)
    a_eq    = np.clip(a.astype(np.int16) + int(NEUTRAL_A - float(a.mean())),
                      0, 255).astype(np.uint8)
    b_eq    = np.clip(b.astype(np.int16) + int(NEUTRAL_B - float(b.mean())),
                      0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_eq, a_eq, b_eq]), cv2.COLOR_LAB2BGR)


# =============================================================================
#  MAKEUP AUGMENTATION  (NEW)
# =============================================================================

def _get_face_landmarks_68(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns 68 dlib-style landmarks if dlib is available, else None.
    We use a simplified mask-based fallback if dlib is not installed.
    """
    try:
        import dlib
        detector   = dlib.get_frontal_face_detector()
        predictor  = dlib.shape_predictor(
            str(Path.home() / ".dlib/shape_predictor_68_face_landmarks.dat"))
        gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        dets  = detector(gray, 1)
        if not dets:
            return None
        shape = predictor(gray, dets[0])
        pts   = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)], dtype=np.float32)
        return pts
    except Exception:
        return None


def _make_lip_mask(h: int, w: int,
                   pts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a soft lip mask for 112x112 face crop.
    Uses landmarks 48-67 if available, else geometric approximation.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    if pts is not None:
        outer = pts[48:60].astype(np.int32)
        inner = pts[60:68].astype(np.int32)
        cv2.fillPoly(mask, [outer], 1.0)
        cv2.fillPoly(mask, [inner], 0.0)
    else:
        # Geometric fallback: lower-third centre strip
        y1 = int(h * 0.68)
        y2 = int(h * 0.85)
        x1 = int(w * 0.30)
        x2 = int(w * 0.70)
        cv2.ellipse(mask, (w//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//2),
                    0, 0, 360, 1.0, -1)
    # Soft edge
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    return mask


def _make_eye_mask(h: int, w: int,
                   pts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a soft eye-area mask for 112x112 face crop.
    Uses landmarks 36-47 if available, else geometric approximation.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    if pts is not None:
        left  = pts[36:42].astype(np.int32)
        right = pts[42:48].astype(np.int32)
        # Expand the eye regions slightly for eyeshadow
        for eye_pts in [left, right]:
            centre = eye_pts.mean(axis=0).astype(int)
            cv2.ellipse(mask, tuple(centre),
                        (int(w*0.14), int(h*0.10)),
                        0, 0, 360, 1.0, -1)
    else:
        # Geometric fallback
        y1 = int(h * 0.33)
        y2 = int(h * 0.55)
        for cx in [int(w*0.30), int(w*0.70)]:
            cv2.ellipse(mask, (cx, (y1+y2)//2),
                        (int(w*0.16), int((y2-y1)//2)),
                        0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 3)
    return mask


def _make_skin_mask(h: int, w: int,
                    pts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a forehead+cheek mask for foundation colour shift.
    Landmarks 0-16 (jaw) define the face outline if available.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    if pts is not None:
        jaw   = pts[0:17].astype(np.int32)
        brow  = pts[17:27].astype(np.int32)
        hull  = np.concatenate([jaw, brow[::-1]], axis=0)
        cv2.fillPoly(mask, [hull], 1.0)
        # Remove eyes and mouth
        if len(pts) >= 68:
            eye_mask = _make_eye_mask(h, w, pts)
            lip_mask = _make_lip_mask(h, w, pts)
            mask     = np.clip(mask - 0.8 * eye_mask - 0.8 * lip_mask, 0, 1)
    else:
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.55)),
                    0, 0, 360, 1.0, -1)
        mask = np.clip(mask
                       - 0.6 * _make_eye_mask(h, w)
                       - 0.6 * _make_lip_mask(h, w), 0, 1)
    mask = cv2.GaussianBlur(mask, (11, 11), 4)
    return mask


def apply_makeup_augmentation(face_bgr: np.ndarray,
                               style: dict) -> np.ndarray:
    """
    Synthetically apply makeup to a face crop:
      1. Foundation — subtle skin-tone warm shift on face region
      2. Eyeshadow  — colour overlay on eye area
      3. Lipstick   — colour overlay on lip area

    All overlays use soft alpha masks so edges are never hard/visible.
    The result looks like lightly applied makeup, not a harsh colour filter.

    Args:
        face_bgr:  112x112 BGR face crop (already aligned + normalised)
        style:     dict with keys 'lip' (BGR tuple), 'eye' (BGR tuple),
                   'foundation' (int brightness shift)

    Returns:
        Augmented 112x112 BGR face crop.
    """
    h, w  = face_bgr.shape[:2]
    out   = face_bgr.astype(np.float32)

    pts   = _get_face_landmarks_68(face_bgr)  # None if dlib unavailable

    # ── Foundation: subtle warm tone shift on skin region ─────────────────
    skin_mask   = _make_skin_mask(h, w, pts)[:, :, np.newaxis]   # (h,w,1)
    foundation  = int(style.get("foundation", 5))
    # Warm shift: raise R slightly, lower B slightly
    warm_shift  = np.array([[-foundation, 0, foundation]], dtype=np.float32)  # BGR
    out         = out + skin_mask * warm_shift
    out         = np.clip(out, 0, 255)

    # ── Eyeshadow: colour overlay on eye region ────────────────────────────
    eye_mask    = _make_eye_mask(h, w, pts)[:, :, np.newaxis]
    eye_color   = np.array(style["eye"], dtype=np.float32).reshape(1, 1, 3)
    eye_alpha   = 0.35                            # blend weight
    out         = out * (1 - eye_alpha * eye_mask) + eye_color * (eye_alpha * eye_mask)
    out         = np.clip(out, 0, 255)

    # ── Lipstick: colour overlay on lip region ─────────────────────────────
    lip_mask    = _make_lip_mask(h, w, pts)[:, :, np.newaxis]
    lip_color   = np.array(style["lip"], dtype=np.float32).reshape(1, 1, 3)
    lip_alpha   = 0.55
    out         = out * (1 - lip_alpha * lip_mask) + lip_color * (lip_alpha * lip_mask)
    out         = np.clip(out, 0, 255)

    return out.astype(np.uint8)


# =============================================================================
#  STEP 1 — FACE ALIGNMENT  +  EYE-DISTANCE CHECK
# =============================================================================

def _load_retinaface():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("RetinaFace (insightface buffalo_sc) loaded")
        return ("retinaface", app)
    except Exception:
        log.warning("insightface not available — trying MTCNN")
        return None


def _load_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn  = MTCNN(keep_all=False, min_face_size=MIN_FACE_PX,
                       thresholds=[0.6, 0.7, 0.7], device=device,
                       post_process=False, select_largest=True)
        log.info(f"MTCNN loaded | device={device}")
        return ("mtcnn", mtcnn)
    except ImportError:
        log.warning("facenet-pytorch not available — using Haar fallback")
        return None


class FaceAligner:
    """
    Detector priority: RetinaFace -> MTCNN -> Haar cascade.
    align() returns (crop, iod).
    """

    def __init__(self, detector=None):
        self.detector = detector
        self.clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cc            = cv2.data.haarcascades
        self.face_cc  = cv2.CascadeClassifier(cc + "haarcascade_frontalface_alt2.xml")
        if self.face_cc.empty():
            self.face_cc = cv2.CascadeClassifier(
                cc + "haarcascade_frontalface_default.xml")
        self.leye_cc  = cv2.CascadeClassifier(cc + "haarcascade_lefteye_2splits.xml")
        self.reye_cc  = cv2.CascadeClassifier(cc + "haarcascade_righteye_2splits.xml")
        log.info(f"FaceAligner | detector={detector[0] if detector else 'Haar only'}")

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        n    = src.shape[0]
        mu_s = src.mean(0);  mu_d = dst.mean(0)
        sc   = src - mu_s;   dc   = dst - mu_d
        vs   = (sc ** 2).sum() / n
        if vs < 1e-10:
            return None
        U, S, Vt = np.linalg.svd((dc.T @ sc) / n)
        d = np.ones(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            d[-1] = -1
        R = U @ np.diag(d) @ Vt
        c = (S * d).sum() / vs
        t = mu_d - c * R @ mu_s
        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = c * R;  M[:, 2] = t
        return M

    def _warp(self, frame: np.ndarray,
              lms: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        iod = float(np.linalg.norm(lms[0] - lms[1]))
        M   = self._umeyama(lms, REFERENCE_PTS)
        if M is None:
            return None, iod
        crop = cv2.warpAffine(frame, M, (FACE_SIZE, FACE_SIZE),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)
        return crop, iod

    def _retinaface_align(self, frame):
        try:
            _, app = self.detector
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces  = app.get(rgb)
            if not faces:
                return None, 0.0
            face = max(faces,
                       key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            return self._warp(frame, face.kps.astype(np.float32))
        except Exception:
            return None, 0.0

    def _mtcnn_align(self, frame):
        try:
            _, mtcnn  = self.detector
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, _, lms = mtcnn.detect(rgb, landmarks=True)
            if lms is not None and len(lms) > 0:
                return self._warp(frame, lms[0].astype(np.float32))
        except Exception:
            pass
        return None, 0.0

    def _haar_align(self, frame):
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        best = None
        for sf, mn in [(1.05, 5), (1.05, 3), (1.10, 2)]:
            faces = self.face_cc.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn,
                minSize=(MIN_FACE_PX, MIN_FACE_PX))
            if len(faces) > 0:
                best = tuple(max(faces, key=lambda f: f[2] * f[3]))
                break
        if best is None:
            return None, 0.0
        x, y, w, h = best
        roi  = gray[y:y + int(h * 0.60), x:x + w]
        half = w // 2

        def _eye(cc, sub, xo, yo):
            for mn in [5, 3, 2]:
                d = cc.detectMultiScale(sub, scaleFactor=1.10, minNeighbors=mn,
                                        minSize=(int(w*0.10), int(w*0.10)))
                if len(d) > 0:
                    ex, ey, ew, eh = max(d, key=lambda d: d[2]*d[3])
                    return np.array([x + xo + ex + ew//2,
                                     y + yo + ey + eh//2], dtype=np.float32)
            return None

        _er = _eye(self.reye_cc, roi[:, :half], 0,    0)
        _el = _eye(self.leye_cc, roi[:, half:], half, 0)
        er  = _er if _er is not None else np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        el  = _el if _el is not None else np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        if er[0] > el[0]:
            er, el = el, er
        lms = np.array([
            er, el,
            [x+0.50*w, y+0.60*h],
            [x+0.35*w, y+0.76*h],
            [x+0.65*w, y+0.76*h],
        ], dtype=np.float32)
        return self._warp(frame, lms)

    def align(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        if self.detector is not None:
            kind = self.detector[0]
            crop, iod = (self._retinaface_align(frame)
                         if kind == "retinaface"
                         else self._mtcnn_align(frame))
            if crop is not None:
                return crop, iod
        return self._haar_align(frame)


def eye_distance_filter(iods: List[float],
                         crops: List[np.ndarray],
                         tol: float = EYE_DIST_TOL
                         ) -> Tuple[List[float], List[np.ndarray]]:
    valid_iods = [d for d in iods if d > 0.0]
    if not valid_iods:
        return iods, crops
    median_iod  = float(np.median(valid_iods))
    kept_iods   = []
    kept_crops  = []
    for iod, crop in zip(iods, crops):
        if iod <= 0.0:
            continue
        if abs(iod - median_iod) / (median_iod + 1e-6) <= tol:
            kept_iods.append(iod)
            kept_crops.append(crop)
    if len(kept_crops) < 5:
        log.warning(
            f"Eye-distance filter would leave {len(kept_crops)} frames — keeping all.")
        return iods, crops
    removed = len(crops) - len(kept_crops)
    if removed > 0:
        log.info(f"  Eye-distance filter removed {removed} frame(s) "
                 f"(median IOD={median_iod:.1f}px, tol={tol*100:.0f}%)")
    return kept_iods, kept_crops


# =============================================================================
#  FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: Path,
                   n_frames: int = FRAMES_TO_USE) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan    = n_frames * CANDIDATE_MULT
    positions = [int(round(i * (total-1) / max(n_scan-1, 1))) for i in range(n_scan)]
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
        raise RuntimeError(f"No frames read: {video_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:n_frames]
    top.sort(key=lambda x: x[1])
    return [f for _, _, f in top]


def resolve_video_path(video_dir: Path, filename: str) -> Path:
    full = video_dir / filename
    if full.exists():
        return full
    fname_lower = filename.lower()
    matches = [p for p in video_dir.iterdir() if p.name.lower() == fname_lower]
    if matches:
        log.warning(f"Case mismatch — using: {matches[0].name}")
        return matches[0]
    raise FileNotFoundError(
        f"Video not found: {full}\n"
        f"Files in directory:\n" + "\n".join(
            f"  {p.name}" for p in sorted(video_dir.iterdir())))


# =============================================================================
#  DATASET BUILDER
# =============================================================================

def build_dataset(video_dir: Path,
                  aligner: FaceAligner,
                  augment: bool = True
                  ) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Extract, align, bilateral-filter, colour-normalise, and optionally
    augment all training frames.

    Returns:
        faces  — list of 112x112 BGR face crops (numpy uint8)
        labels — list of integer class ids (0 = V8, 1 = V9, ...)
        info   — list of descriptive strings for logging
    """
    faces:  List[np.ndarray] = []
    labels: List[int]        = []
    info:   List[str]        = []

    person_to_id = {p: i for i, p in enumerate(PERSONS)}

    for person, variant, filename in VIDEOS:
        vpath    = resolve_video_path(video_dir, filename)
        label_id = person_to_id[person]
        tag      = f"{person}_{variant}"

        log.info(f"Extracting  {tag}")
        frames = extract_frames(vpath, n_frames=FRAMES_TO_USE)

        raw_iods:  List[float]      = []
        raw_crops: List[np.ndarray] = []

        for frame in frames:
            aligned, iod = aligner.align(frame)
            if aligned is None:
                continue
            smoothed  = bilateral_smooth(aligned)
            normed    = normalise_face(smoothed)
            raw_iods.append(iod)
            raw_crops.append(normed)

        if not raw_crops:
            log.error(f"No faces detected in {tag} — skipping")
            continue

        _, filt_crops = eye_distance_filter(raw_iods, raw_crops)
        log.info(f"  {tag:<35}  kept {len(filt_crops):>3}/{len(frames)} frames")

        for crop in filt_crops:
            faces.append(crop)
            labels.append(label_id)
            info.append(f"{tag}_real")

        # ── Makeup augmentation on no_makeup frames only ──────────────────
        if augment and variant == "no_makeup":
            for style_idx, style in enumerate(MAKEUP_STYLES):
                for crop in filt_crops:
                    aug = apply_makeup_augmentation(crop, style)
                    faces.append(aug)
                    labels.append(label_id)
                    info.append(f"{tag}_aug{style_idx}")
            log.info(f"  {tag:<35}  added {len(filt_crops)*len(MAKEUP_STYLES):>3} "
                     f"augmented frames ({len(MAKEUP_STYLES)} styles)")

    log.info(f"\nDataset: {len(faces)} total samples  |  "
             f"{NUM_CLASSES} classes  |  "
             f"augmented={augment}")
    return faces, labels, info


# =============================================================================
#  HARD TRIPLET MINING  (NEW)
# =============================================================================

def mine_hard_triplets(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    margin:     float = TRIPLET_MARGIN,
    strategy:   str   = "semi_hard",
) -> List[Tuple[int, int, int]]:
    """
    Online hard/semi-hard triplet mining.

    For each anchor i:
      - Hard positive: same class, LOWEST cosine similarity (most different looking)
      - Hard negative: different class, HIGHEST cosine similarity (most confusable)
      - Semi-hard negative: different class, sim(a,n) > sim(a,p) - margin (violators)

    Args:
        embeddings: (N, 512) L2-normalised feature matrix
        labels:     (N,) integer class labels
        margin:     triplet margin
        strategy:   "hard" | "semi_hard"
                    hard     — always use the single hardest negative
                    semi_hard — prefer semi-hard violators (more stable training)

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) tuples.
        Only valid triplets (loss > 0) are returned.
    """
    # Cosine similarity matrix — fast via dot product on L2-norm embeddings
    sim_matrix = embeddings @ embeddings.T   # (N, N)
    n          = len(labels)
    triplets   = []

    for i in range(n):
        label_i   = labels[i]
        sim_row   = sim_matrix[i]

        pos_mask  = (labels == label_i)
        pos_mask[i] = False
        neg_mask  = (labels != label_i)

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Hard positive: same class, min similarity
        pos_sims  = np.where(pos_mask, sim_row, np.inf)
        hard_pos  = int(np.argmin(pos_sims))
        sim_ap    = sim_row[hard_pos]

        if strategy == "semi_hard":
            # Semi-hard: negatives harder than positive but within margin
            #   sim(a,n) > sim(a,p) - margin
            semi_hard_neg_mask = neg_mask & (sim_row > sim_ap - margin)
            if semi_hard_neg_mask.any():
                # Among semi-hard negatives, take the hardest (highest sim)
                neg_sims = np.where(semi_hard_neg_mask, sim_row, -np.inf)
                hard_neg = int(np.argmax(neg_sims))
            else:
                # Fall back to hardest negative
                neg_sims = np.where(neg_mask, sim_row, -np.inf)
                hard_neg = int(np.argmax(neg_sims))
        else:
            # Pure hard: highest-similarity negative
            neg_sims = np.where(neg_mask, sim_row, -np.inf)
            hard_neg = int(np.argmax(neg_sims))

        sim_an = sim_row[hard_neg]

        # Only keep triplet if there is loss (push beyond margin)
        loss = max(0.0, (1 - sim_ap) - (1 - sim_an) + margin)
        if loss > 0:
            triplets.append((i, hard_pos, hard_neg))

    return triplets


def compute_triplet_loss_np(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    margin:     float = TRIPLET_MARGIN,
) -> float:
    """
    Compute mean triplet loss over all mined hard/semi-hard triplets.
    Pure NumPy — used for monitoring when PyTorch is unavailable.
    """
    triplets = mine_hard_triplets(embeddings, labels, margin)
    if not triplets:
        return 0.0
    losses = []
    for a, p, n in triplets:
        dist_pos = 1.0 - float(np.dot(embeddings[a], embeddings[p]))
        dist_neg = 1.0 - float(np.dot(embeddings[a], embeddings[n]))
        losses.append(max(0.0, dist_pos - dist_neg + margin))
    return float(np.mean(losses)) if losses else 0.0


# =============================================================================
#  SUB-CENTER ADAFACE LOSS  (PyTorch)
# =============================================================================

def _build_subcenter_adaface_head():
    """
    Returns the SubCenter AdaFace head as a PyTorch nn.Module.
    If PyTorch is not installed, returns None and training falls back
    to a NumPy SGD approximation.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SubCenterAdaFace(nn.Module):
            """
            AdaFace margin loss with sub-center support.

            Each class has K=3 prototype vectors (sub-centers).
            During forward, the feature is matched to the nearest sub-center
            of its assigned class, reducing intra-class variance from makeup.

            Reference: AdaFace (Kim et al. 2022) + Sub-center ArcFace (Deng et al. 2020)
            """

            def __init__(self,
                         embedding_size: int = 512,
                         num_classes:    int = NUM_CLASSES,
                         K:              int = K_SUBCENTERS,
                         m:              float = 0.4,
                         h:              float = 0.333,
                         s:              float = 64.0,
                         t_alpha:        float = 0.01):
                super().__init__()
                self.embedding_size = embedding_size
                self.num_classes    = num_classes
                self.K              = K
                self.m              = m
                self.h              = h
                self.s              = s
                self.t_alpha        = t_alpha
                self.eps            = 1e-3

                # Weight tensor: (num_classes * K, embedding_size)
                # Each block of K rows = K sub-centers for one class
                self.W = nn.Parameter(
                    torch.Tensor(num_classes * K, embedding_size))
                nn.init.xavier_uniform_(self.W)

                # Running mean of ||z|| (feature norm) for adaptive margin
                self.register_buffer(
                    "norm_mean",
                    torch.ones(1) * 20.0)

            def _get_subcenter_logits(self, features_norm: "torch.Tensor"
                                      ) -> "torch.Tensor":
                """
                Compute cosine similarity between each sample and ALL
                K*C sub-centers, then for each class take the MAX
                (nearest sub-center).  Output: (batch, num_classes).
                """
                W_norm = F.normalize(self.W, dim=1)
                # (batch, K*C)
                all_cos = features_norm @ W_norm.T
                # Reshape to (batch, C, K)
                all_cos = all_cos.view(-1, self.num_classes, self.K)
                # Nearest sub-center for each class
                cos_theta, _ = all_cos.max(dim=2)  # (batch, C)
                return cos_theta

            def forward(self,
                        features:     "torch.Tensor",
                        norms:        "torch.Tensor",
                        labels:       "torch.Tensor") -> "torch.Tensor":
                """
                Args:
                    features: (B, 512) L2-normalised embeddings
                    norms:    (B,)     raw feature norms (before L2)
                    labels:   (B,)     integer class labels

                Returns:
                    Scalar cross-entropy AdaFace loss.
                """
                import torch

                features_norm = F.normalize(features, dim=1)
                cos_theta     = self._get_subcenter_logits(features_norm)
                cos_theta      = cos_theta.clamp(-1 + self.eps, 1 - self.eps)

                # Update running mean of feature norm
                batch_mean = norms.mean().detach()
                self.norm_mean = (
                    (1 - self.t_alpha) * self.norm_mean
                    + self.t_alpha * batch_mean
                )

                # Adaptive margin: scale by normalised norm
                norm_margin = (norms / (self.norm_mean + self.eps)).clamp(0.01, 5)

                # AdaFace: g_angular + g_additive
                theta          = torch.acos(cos_theta)
                # Safe indicator: targets
                one_hot        = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

                g_angular      = self.m * norm_margin.unsqueeze(1) * one_hot
                g_additive     = self.h * norm_margin.unsqueeze(1) * one_hot

                theta_m        = theta + g_angular
                cos_theta_m    = torch.cos(theta_m)
                logits         = cos_theta - g_additive
                logits         = torch.where(
                    one_hot.bool(), cos_theta_m - g_additive, logits)
                logits         = self.s * logits

                loss = F.cross_entropy(logits, labels)
                return loss

        return SubCenterAdaFace

    except ImportError:
        return None


# =============================================================================
#  ADAFACE IR-18 BACKBONE  (PyTorch or ONNX)
# =============================================================================

def _build_ir18_pytorch():
    """
    Build AdaFace IR-18 backbone in PyTorch.
    Loads pretrained weights from WEIGHTS_PATH if it is a .pth file.
    Returns (model, device) or None if PyTorch unavailable.
    """
    try:
        import torch
        import torch.nn as nn

        # Minimal iResNet-18 compatible with AdaFace
        class IBasicBlock(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                self.bn1   = nn.BatchNorm2d(in_c)
                self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
                self.bn2   = nn.BatchNorm2d(out_c)
                self.prelu = nn.PReLU(out_c)
                self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
                self.bn3   = nn.BatchNorm2d(out_c)
                self.down  = None
                if stride != 1 or in_c != out_c:
                    self.down = nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                        nn.BatchNorm2d(out_c))

            def forward(self, x):
                r = self.bn1(x)
                r = self.conv1(r)
                r = self.bn2(r)
                r = self.prelu(r)
                r = self.conv2(r)
                r = self.bn3(r)
                s = self.down(x) if self.down else x
                return r + s

        class IResNet18(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                self.bn1   = nn.BatchNorm2d(64)
                self.prelu = nn.PReLU(64)
                self.layer1 = self._make(64,  64,  2, 1)
                self.layer2 = self._make(64,  128, 2, 2)
                self.layer3 = self._make(128, 256, 2, 2)
                self.layer4 = self._make(256, 512, 2, 2)
                self.bn2    = nn.BatchNorm2d(512)
                self.drop   = nn.Dropout(0.4)
                self.fc     = nn.Linear(512 * 4 * 4, 512)
                self.features = nn.BatchNorm1d(512)
                nn.init.constant_(self.features.weight, 1.0)
                self.features.weight.requires_grad = False

            @staticmethod
            def _make(in_c, out_c, blocks, stride):
                layers = [IBasicBlock(in_c, out_c, stride)]
                for _ in range(1, blocks):
                    layers.append(IBasicBlock(out_c, out_c, 1))
                return nn.Sequential(*layers)

            def forward(self, x):
                # x: (B, 3, 112, 112) normalised to [-1, 1]
                x = self.prelu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.bn2(x)
                x = self.drop(x)
                x = x.flatten(1)
                norm = x.norm(dim=1, keepdim=True)
                x = self.fc(x)
                x = self.features(x)
                return x, norm.squeeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = IResNet18().to(device)

        if Path(WEIGHTS_PATH).exists() and WEIGHTS_PATH.endswith(".pth"):
            state = torch.load(WEIGHTS_PATH, map_location=device)
            # Handle wrapped state dicts
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state:
                state = state["model"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log.warning(f"Missing keys: {missing[:5]} ...")
            log.info(f"IR-18 weights loaded from {WEIGHTS_PATH}")
        else:
            log.warning(
                f"Weights not found or not .pth: {WEIGHTS_PATH}\n"
                "Training with random initialisation — "
                "recommend converting ONNX weights first.")

        log.info(f"IR-18 backbone | device={device}")
        return model, device

    except ImportError:
        return None


class OnnxEncoder:
    """Fallback: ONNX-only encoder for embedding extraction (no training)."""

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
        log.info("OnnxEncoder loaded (inference only — cannot fine-tune)")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        inp = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: inp})
        raw = out[0][0] if out[0].ndim == 2 else out[0]
        n   = np.linalg.norm(raw)
        return (raw / n).astype(np.float32) if n > 1e-10 else raw.astype(np.float32)

    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(c) for c in crops])


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-10 else v.astype(np.float32)


# =============================================================================
#  PYTORCH TRAINING LOOP
# =============================================================================

def train_pytorch(
    model_and_device: tuple,
    faces:  List[np.ndarray],
    labels: List[int],
) -> Dict[str, list]:
    """
    Full PyTorch fine-tuning loop.

    Loss = SubCenter AdaFace loss  +  TRIPLET_WEIGHT * Triplet loss

    Strategy:
      1. Freeze all backbone layers except layer4 + FC + BN for first 5 epochs
         (head-only warm-up prevents catastrophic forgetting on 6 identities)
      2. Unfreeze full backbone from epoch 6 onward
      3. Cosine LR schedule

    Returns dict with training history.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    model, device = model_and_device

    SubCenterAdaFace = _build_subcenter_adaface_head()
    if SubCenterAdaFace is None:
        raise RuntimeError("PyTorch not available for training")
    head = SubCenterAdaFace().to(device)

    class FaceDataset(Dataset):
        def __init__(self, crops, lbls):
            self.crops = crops
            self.lbls  = lbls

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            img = self.crops[idx]
            img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
            img = torch.from_numpy(img.transpose(2, 0, 1))
            return img, self.lbls[idx]

    dataset    = FaceDataset(faces, labels)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=False, num_workers=0)
    label_arr  = np.array(labels)

    # ── Phase 1: freeze backbone except layer4+head (epochs 1-5) ─────────
    def set_trainable(epoch):
        if epoch < 5:
            for name, param in model.named_parameters():
                param.requires_grad = any(
                    k in name for k in ["layer4", "bn2", "drop", "fc", "features"])
        else:
            for param in model.parameters():
                param.requires_grad = True

    optimizer  = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"epoch": [], "ada_loss": [], "trip_loss": [], "total_loss": [],
               "within_sim_mean": [], "lr": []}

    log.info(f"\nStarting fine-tuning: {NUM_EPOCHS} epochs  |  "
             f"batch={BATCH_SIZE}  |  lr={LEARNING_RATE}")

    for epoch in range(NUM_EPOCHS):
        set_trainable(epoch)
        model.train()
        head.train()

        epoch_ada   = []
        epoch_trip  = []
        epoch_total = []

        for batch_imgs, batch_labels in loader:
            batch_imgs   = batch_imgs.to(device)
            batch_labels = batch_labels.to(device).long()

            feats, norms = model(batch_imgs)

            # ── AdaFace sub-center loss ────────────────────────────────────
            ada_loss = head(feats, norms, batch_labels)

            # ── Triplet loss ───────────────────────────────────────────────
            with torch.no_grad():
                feats_np  = F.normalize(feats, dim=1).cpu().numpy()
                lbls_np   = batch_labels.cpu().numpy()

            triplets  = mine_hard_triplets(feats_np, lbls_np, margin=TRIPLET_MARGIN,
                                           strategy="semi_hard")
            trip_loss = torch.tensor(0.0, device=device)
            if triplets:
                a_idx = torch.tensor([t[0] for t in triplets], device=device)
                p_idx = torch.tensor([t[1] for t in triplets], device=device)
                n_idx = torch.tensor([t[2] for t in triplets], device=device)
                feats_n = F.normalize(feats, dim=1)
                d_pos   = 1.0 - (feats_n[a_idx] * feats_n[p_idx]).sum(dim=1)
                d_neg   = 1.0 - (feats_n[a_idx] * feats_n[n_idx]).sum(dim=1)
                trip_loss = F.relu(d_pos - d_neg + TRIPLET_MARGIN).mean()

            total_loss = ada_loss + TRIPLET_WEIGHT * trip_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 5.0)
            optimizer.step()

            epoch_ada.append(ada_loss.item())
            epoch_trip.append(trip_loss.item() if isinstance(trip_loss, torch.Tensor)
                               else float(trip_loss))
            epoch_total.append(total_loss.item())

        scheduler.step()

        # ── Within-identity evaluation ─────────────────────────────────────
        model.eval()
        with torch.no_grad():
            all_embs: Dict[str, np.ndarray] = {}
            for person in PERSONS:
                for variant in ["makeup", "no_makeup"]:
                    key = f"{person}_{variant}"
                    idx = [i for i, inf in enumerate(
                        [f"{p}_{v}_real" for p, v, _ in VIDEOS
                         for _ in range(1)]) if inf.startswith(key)]
                    # Rough: grab faces matching this person+variant
                    p_faces = [f for f, l in zip(faces, labels)
                               if l == PERSONS.index(person)]
                    if not p_faces:
                        continue
                    vecs = []
                    for f in p_faces[:20]:
                        img = cv2.resize(f, (FACE_SIZE, FACE_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
                        t   = torch.from_numpy(
                            img.transpose(2, 0, 1)).unsqueeze(0).to(device)
                        emb, _ = model(t)
                        emb = F.normalize(emb, dim=1).cpu().numpy()[0]
                        vecs.append(emb)
                    all_embs[key] = _l2(np.stack(vecs).mean(axis=0))

        within_sims = []
        for person in PERSONS:
            mk = all_embs.get(f"{person}_makeup")
            nm = all_embs.get(f"{person}_no_makeup")
            if mk is not None and nm is not None:
                within_sims.append(float(np.dot(mk, nm)))

        mean_ada   = float(np.mean(epoch_ada))
        mean_trip  = float(np.mean(epoch_trip))
        mean_total = float(np.mean(epoch_total))
        mean_wsim  = float(np.mean(within_sims)) if within_sims else 0.0
        cur_lr     = float(scheduler.get_last_lr()[0])

        history["epoch"].append(epoch + 1)
        history["ada_loss"].append(mean_ada)
        history["trip_loss"].append(mean_trip)
        history["total_loss"].append(mean_total)
        history["within_sim_mean"].append(mean_wsim)
        history["lr"].append(cur_lr)

        log.info(
            f"  Epoch {epoch+1:>3}/{NUM_EPOCHS}  "
            f"ada={mean_ada:.4f}  trip={mean_trip:.4f}  "
            f"total={mean_total:.4f}  "
            f"within_sim={mean_wsim:.4f}  lr={cur_lr:.2e}"
        )

    return model, head, history


# =============================================================================
#  EVAL — 12x12 COSINE MATRIX + WITHIN-IDENTITY REPORT
# =============================================================================

def eval_pipeline(model_and_device,
                  faces: List[np.ndarray],
                  labels_list: List[int],
                  use_onnx: bool = False,
                  onnx_encoder: Optional[OnnxEncoder] = None
                  ) -> None:
    """
    Compute final cosine similarity matrix after fine-tuning.
    Mirrors the output format of the inference pipeline.
    """
    import torch
    import torch.nn.functional as F

    model, device = model_and_device
    model.eval()

    person_embeddings: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for person_idx, person in enumerate(PERSONS):
            for variant in ["makeup", "no_makeup"]:
                key   = f"{person}_{variant}"
                p_faces = [f for f, l in zip(faces, labels_list)
                           if l == person_idx]
                if not p_faces:
                    log.warning(f"No faces for {key} — skipping")
                    continue
                vecs = []
                for face in p_faces[:FRAMES_TO_USE]:
                    img = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
                    t   = torch.from_numpy(
                        img.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    emb, _ = model(t)
                    emb = F.normalize(emb, dim=1).cpu().numpy()[0]
                    # H-flip
                    img_f = np.ascontiguousarray(img[:, :, ::-1])
                    tf    = torch.from_numpy(
                        img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    emb_f, _ = model(tf)
                    emb_f = F.normalize(emb_f, dim=1).cpu().numpy()[0]
                    vecs.append(_l2(emb + emb_f))

                # Sub-center aggregate
                arr    = np.stack(vecs)
                k      = min(K_SUBCENTERS, len(arr))
                if k > 1:
                    km     = KMeans(n_clusters=k, n_init=10, random_state=42)
                    lbls   = km.fit_predict(arr)
                    best   = None
                    best_s = -np.inf
                    for c in range(k):
                        mem = arr[lbls == c]
                        if not len(mem):
                            continue
                        cen = _l2(mem.mean(axis=0))
                        s   = float((mem @ cen).mean())
                        if s > best_s:
                            best_s = s
                            best   = cen
                    person_embeddings[key] = best if best is not None else _l2(arr.mean(0))
                else:
                    person_embeddings[key] = _l2(arr.mean(0))

    ordered_keys = [f"{p}_{v}" for p, _, _ in VIDEOS
                    for v in []] + \
                   [f"{p}_{v}" for p in PERSONS
                    for v in ["makeup", "no_makeup"]]
    # De-duplicate while preserving order
    seen = set()
    keys = []
    for p in PERSONS:
        for v in ["makeup", "no_makeup"]:
            k = f"{p}_{v}"
            if k not in seen and k in person_embeddings:
                keys.append(k)
                seen.add(k)

    n  = len(keys)
    sm = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sm[i, j] = float(np.dot(person_embeddings[keys[i]],
                                    person_embeddings[keys[j]]))

    def short(k):
        p, v = k.rsplit("_", 1)
        return f"{p}-{'MU' if v == 'makeup' else 'NM'}"

    slabels = [short(k) for k in keys]

    SEP  = "=" * 72
    SEP2 = "-" * 72

    within_scores = []
    print(f"\n{SEP}")
    print("  POST-FINETUNE WITHIN-IDENTITY SIMILARITIES")
    print(SEP)
    print(f"  {'Pair':<42}  {'Cosine Sim':>10}")
    print(f"  {'-'*42}  {'-'*10}")
    for person in PERSONS:
        mk = f"{person}_makeup"
        nm = f"{person}_no_makeup"
        if mk in person_embeddings and nm in person_embeddings:
            s = float(np.dot(person_embeddings[mk], person_embeddings[nm]))
            within_scores.append(s)
            print(f"  {person+' makeup':<20}  vs  {person+' no makeup':<18}  {s:>10.4f}")

    passing = sum(1 for s in within_scores if s >= THRESHOLD)
    print(f"\n  Mean   : {np.mean(within_scores):.4f}")
    print(f"  Min    : {np.min(within_scores):.4f}")
    print(f"  Max    : {np.max(within_scores):.4f}")
    print(f"  Passing: {passing}/{len(within_scores)} above {THRESHOLD}")

    print(f"\n{SEP}")
    print("  FULL 12x12 COSINE SIMILARITY MATRIX  (post fine-tune)")
    print(SEP)
    col_w  = 8
    header = " " * 10 + "".join(f"{s:>{col_w}}" for s in slabels)
    print(f"  {header}")
    print(f"  {'-'*(10 + col_w*n)}")
    for i, rl in enumerate(slabels):
        row = f"  {rl:<8}"
        for j in range(n):
            row += f"{sm[i,j]:>{col_w}.4f}"
        print(row)

    # Cross-identity pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pi    = keys[i].split("_")[0]
            pj    = keys[j].split("_")[0]
            ptype = "same" if pi == pj else "diff"
            pairs.append((slabels[i], slabels[j], sm[i,j], ptype))
    pairs.sort(key=lambda x: x[2], reverse=True)

    cross = [s for _, _, s, t in pairs if t == "diff"]
    print(f"\n{SEP}")
    print("  SUMMARY  (post fine-tune)")
    print(SEP)
    print(f"  Within-identity mean  : {np.mean(within_scores):.4f}")
    print(f"  Cross-identity  mean  : {np.mean(cross):.4f}")
    gap = np.mean(within_scores) - np.mean(cross)
    print(f"  Separation gap        : {gap:.4f}  "
          f"({'Good' if gap > 0.15 else 'Low'})")
    print(f"  Passing (>={THRESHOLD}) : {passing}/{len(within_scores)}")
    print(f"\n{SEP}")

    return person_embeddings, within_scores


# =============================================================================
#  SAVE RESULTS
# =============================================================================

def save_results(model_and_device, head, history: dict,
                 out_dir: Path) -> None:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = model_and_device

    # Save model weights
    pth_path = out_dir / "finetuned_adaface_ir18.pth"
    torch.save(model.state_dict(), pth_path)
    log.info(f"Model saved: {pth_path}")

    # Save training log
    csv_path = out_dir / "finetune_log.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,ada_loss,trip_loss,total_loss,within_sim_mean,lr\n")
        for i, ep in enumerate(history["epoch"]):
            f.write(
                f"{ep},{history['ada_loss'][i]:.6f},"
                f"{history['trip_loss'][i]:.6f},"
                f"{history['total_loss'][i]:.6f},"
                f"{history['within_sim_mean'][i]:.6f},"
                f"{history['lr'][i]:.8f}\n"
            )
    log.info(f"Training log saved: {csv_path}")


# =============================================================================
#  NUMPY FALLBACK TRAINING (no PyTorch)
# =============================================================================

def train_numpy_fallback(
    encoder:    OnnxEncoder,
    faces:      List[np.ndarray],
    labels_arr: List[int],
) -> Dict:
    """
    Approximate training monitoring without gradient descent.
    Extracts embeddings with the ONNX encoder and reports:
      - Baseline within-identity similarities
      - Hard triplet loss (as diagnostic, no weight update)
      - Sub-center coherence per class

    This is a fallback for when PyTorch is unavailable.
    Fine-tuning requires PyTorch; this only provides diagnostics.
    """
    log.warning(
        "PyTorch not available — running ONNX diagnostic mode (no weight updates)")

    embs = encoder.embed_batch(faces)
    embs = np.stack([_l2(e) for e in embs])
    lbl  = np.array(labels_arr)

    trip_loss = compute_triplet_loss_np(embs, lbl)
    log.info(f"  Baseline triplet loss  : {trip_loss:.4f}")

    within_sims = []
    for person_idx, person in enumerate(PERSONS):
        mask = lbl == person_idx
        if not mask.any():
            continue
        cen  = _l2(embs[mask].mean(axis=0))
        coh  = float((embs[mask] @ cen).mean())
        log.info(f"  {person:<6} cluster coherence: {coh:.4f}")

    log.info("\nTo enable fine-tuning: pip install torch torchvision")
    return {"diagnostic_only": True, "baseline_triplet_loss": trip_loss}


# =============================================================================
#  MAIN
# =============================================================================

SEP  = "=" * 72
SEP2 = "-" * 72


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = _load_retinaface() or _load_mtcnn()
    aligner  = FaceAligner(detector=detector)

    print(f"\n{SEP}")
    print("  ADAFACE FINE-TUNING PIPELINE")
    print("  Sub-center Loss  +  Makeup Augmentation  +  Hard Triplet Mining")
    print(SEP)
    print(f"  Video directory : {VIDEO_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Detector        : {detector[0] if detector else 'Haar only'}")
    print(f"  Frames/video    : {FRAMES_TO_USE} (training) + {len(MAKEUP_STYLES)} aug styles")
    print(f"  Sub-centers K   : {K_SUBCENTERS}")
    print(f"  Triplet margin  : {TRIPLET_MARGIN}  weight={TRIPLET_WEIGHT}")
    print(f"  Epochs          : {NUM_EPOCHS}  batch={BATCH_SIZE}  lr={LEARNING_RATE}")
    print(SEP2)

    # ── Build dataset ──────────────────────────────────────────────────────
    faces, labels_list, info = build_dataset(VIDEO_DIR, aligner, augment=True)

    # ── Try PyTorch training ───────────────────────────────────────────────
    backbone = _build_ir18_pytorch()

    if backbone is not None:
        model, device = backbone
        trained_model, head, history = train_pytorch(
            (model, device), faces, labels_list)
        save_results((trained_model, device), head, history, OUTPUT_DIR)
        eval_pipeline((trained_model, device), faces, labels_list)
    else:
        # ONNX fallback — diagnostics only
        log.warning("Falling back to ONNX encoder (diagnostics only)")
        onnx_path = WEIGHTS_PATH if Path(WEIGHTS_PATH).exists() else None
        if onnx_path is None:
            raise FileNotFoundError(f"No weights found at: {WEIGHTS_PATH}")
        encoder = OnnxEncoder(onnx_path)
        train_numpy_fallback(encoder, faces, labels_list)

    print(f"\n{SEP}")
    print("  FINE-TUNING COMPLETE")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print(f"    finetuned_adaface_ir18.pth  — fine-tuned backbone weights")
    print(f"    finetune_log.csv             — per-epoch loss history")
    print(SEP)


if __name__ == "__main__":
    main()
