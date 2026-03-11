"""
AdaFace High-Similarity Pipeline — Bilateral Filter Edition
===========================================================
Base: Run 1 (best results so far — Haar detector, colour normalisation, K=3)
New:  Bilateral filter applied to aligned crop BEFORE colour normalisation.

Bilateral filter smooths foundation/powder texture noise while preserving
hard edges from facial geometry (bone ridges, nose bridge, jaw). This
directly targets the makeup problem — fine texture from foundation makes
the same face look different to the model across makeup/no-makeup conditions.

Pipeline per video (top-20 sharpest frames):

  Step 1  — Face Alignment (RetinaFace -> MTCNN fallback -> Haar fallback)
             5-landmark Umeyama warp -> 112x112 crop
             Eye-distance consistency check: rejects crops where IOD
             deviates more than 20% from per-video median

  Step 1b — Bilateral Filter  (NEW)
             cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)
             Applied BEFORE colour normalisation on the raw aligned crop.
             Smooths makeup texture (foundation, powder, blush) while
             keeping structural edges (bone ridges, nose, jaw) intact.

  Step 2  — Colour Normalisation
             a. Gamma correction -> target luminance 130
             b. LAB CLAHE on L channel
             c. A/B channel mean-shift to neutral (removes foundation cast)

  Step 3  — Sub-center Aggregation  (K=3)
             Embed 20 frames -> KMeans K=3 -> pick most cohesive cluster.

  Step 4  — AdaFace IR-18 (ONNX)
             embed(norm) + embed(H-flip) -> L2-norm -> frame vector
             Average frame vectors -> L2-renorm -> final unit vector

Output:
  - Within-identity pairs (makeup vs no-makeup)
  - Full 12x12 cosine similarity matrix
  - All pairwise similarities sorted high -> low
  - Summary with separation gap
"""

import logging
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
WEIGHTS_PATH = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"

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

FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112
MIN_FACE_PX    = 40
K_SUBCENTERS   = 3
THRESHOLD      = 0.75
EYE_DIST_TOL   = 0.20

# Bilateral filter parameters
# d=9        : diameter of pixel neighbourhood (good for 112x112 face)
# sigmaColor : colours within 75 units treated as similar -> texture smoothed
# sigmaSpace : spatial influence radius
BILATERAL_D           = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

TARGET_LUMINANCE = 130.0
LAB_CLAHE        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
NEUTRAL_A        = 128.0
NEUTRAL_B        = 128.0

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

    Weights neighbours by spatial distance AND colour similarity:
      - Foundation / powder / blush (fine uniform texture) -> smoothed away
      - Facial structure edges (nose bridge, jaw, cheekbones) -> preserved

    Applied BEFORE colour normalisation so CLAHE works on already-smoothed
    skin rather than amplifying makeup texture noise.
    """
    return cv2.bilateralFilter(
        face_bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )


# =============================================================================
#  STEP 2 — COLOUR NORMALISATION
# =============================================================================

def normalise_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Colour normalisation on the bilateral-smoothed aligned crop:
      a. Gamma correction -> target luminance 130
      b. LAB CLAHE on L channel (local contrast equalisation)
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
        lut   = np.array(
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
    align() returns (crop, iod) where iod is the raw inter-ocular distance
    from detected landmarks before warping.
    """

    def __init__(self, detector=None):
        self.detector  = detector
        self.clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cc             = cv2.data.haarcascades
        self.face_cc   = cv2.CascadeClassifier(cc + "haarcascade_frontalface_alt2.xml")
        if self.face_cc.empty():
            self.face_cc = cv2.CascadeClassifier(
                cc + "haarcascade_frontalface_default.xml")
        self.leye_cc   = cv2.CascadeClassifier(cc + "haarcascade_lefteye_2splits.xml")
        self.reye_cc   = cv2.CascadeClassifier(cc + "haarcascade_righteye_2splits.xml")
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
        iod  = float(np.linalg.norm(lms[0] - lms[1]))
        M    = self._umeyama(lms, REFERENCE_PTS)
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
            _, mtcnn = self.detector
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                                        minSize=(int(w * 0.10), int(w * 0.10)))
                if len(d) > 0:
                    ex, ey, ew, eh = max(d, key=lambda d: d[2] * d[3])
                    return np.array([x + xo + ex + ew // 2,
                                     y + yo + ey + eh // 2], dtype=np.float32)
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
                         vecs: List[np.ndarray],
                         tol:  float = EYE_DIST_TOL
                         ) -> Tuple[List[float], List[np.ndarray]]:
    """
    Remove frames whose IOD deviates more than tol from the per-video median.
    Safety floor: never reduce below 5 frames.
    """
    valid_iods = [d for d in iods if d > 0.0]
    if not valid_iods:
        return iods, vecs

    median_iod = float(np.median(valid_iods))
    kept_iods: List[float]      = []
    kept_vecs: List[np.ndarray] = []

    for iod, vec in zip(iods, vecs):
        if iod <= 0.0:
            continue
        if abs(iod - median_iod) / (median_iod + 1e-6) <= tol:
            kept_iods.append(iod)
            kept_vecs.append(vec)

    if len(kept_vecs) < 5:
        log.warning(
            f"Eye-distance filter would leave only {len(kept_vecs)} frames "
            f"(median IOD={median_iod:.1f}px) — keeping all {len(vecs)} frames.")
        return iods, vecs

    removed = len(vecs) - len(kept_vecs)
    if removed > 0:
        log.info(f"  Eye-distance filter removed {removed} frame(s) "
                 f"(median IOD={median_iod:.1f}px, tol={tol*100:.0f}%)")
    return kept_iods, kept_vecs


# =============================================================================
#  STEP 4 — ADAFACE MODEL
# =============================================================================

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

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Returns L2-normalised 512-dim embedding."""
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        inp = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: inp})
        raw = out[0][0] if out[0].ndim == 2 else out[0]
        n   = np.linalg.norm(raw)
        return (raw / n).astype(np.float32) if n > 1e-10 else raw.astype(np.float32)


# =============================================================================
#  FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan    = FRAMES_TO_USE * CANDIDATE_MULT
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
    top = candidates[:FRAMES_TO_USE]
    top.sort(key=lambda x: x[1])
    return [f for _, _, f in top]


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-10 else v.astype(np.float32)


# =============================================================================
#  STEP 3 — SUB-CENTER AGGREGATION  (K=3)
# =============================================================================

def subcenter_aggregate(frame_vecs: np.ndarray,
                         k: int = K_SUBCENTERS) -> np.ndarray:
    """
    Cluster N frame vectors into K=3 sub-centers (KMeans).
    Select the cluster with highest mean intra-cluster cosine similarity
    as the identity representative.
    """
    n = len(frame_vecs)
    if n <= k:
        return _l2(frame_vecs.mean(axis=0))

    km     = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(frame_vecs)

    best_center = None
    best_score  = -np.inf

    for c in range(k):
        members = frame_vecs[labels == c]
        if len(members) == 0:
            continue
        centroid = _l2(members.mean(axis=0))
        score    = float((members @ centroid).mean())
        if score > best_score:
            best_score  = score
            best_center = centroid

    return best_center if best_center is not None else _l2(frame_vecs.mean(axis=0))


# =============================================================================
#  FULL PIPELINE PER VIDEO
# =============================================================================

def embed_video(video_path: Path,
                model: AdaFaceModel,
                aligner: FaceAligner,
                label: str) -> Optional[np.ndarray]:
    """
    Full pipeline for one video:
      Step 1  — Align + IOD capture
      Step 1b — Bilateral filter  (texture smoothing, edge preserving)
      Step 2  — Colour normalisation
      Step 4  — AdaFace embed + H-flip
      Step 1c — IOD filter
      Step 3  — Sub-center aggregation K=3
    """
    frames   = extract_frames(video_path)
    all_iods: List[float]      = []
    all_vecs: List[np.ndarray] = []

    for frame in frames:
        # Step 1 — Align + capture IOD
        aligned, iod = aligner.align(frame)
        if aligned is None:
            continue

        # Step 1b — Bilateral filter: smooth makeup texture, keep edges
        smoothed = bilateral_smooth(aligned)

        # Step 2 — Colour normalisation on smoothed crop
        norm_face = normalise_face(smoothed)

        # Step 4 — AdaFace embedding: orig + H-flip
        e_orig = model.embed(norm_face)
        e_flip = model.embed(cv2.flip(norm_face, 1))
        vec    = _l2(e_orig + e_flip)

        all_iods.append(iod)
        all_vecs.append(vec)

    if not all_vecs:
        log.error(f"No faces detected: {Path(video_path).name}")
        return None

    # Step 1c — Eye-distance consistency filter
    _, filt_vecs = eye_distance_filter(all_iods, all_vecs)

    vecs_array = np.stack(filt_vecs)

    # Step 3 — Sub-center aggregation K=3
    final = subcenter_aggregate(vecs_array, k=K_SUBCENTERS)

    log.info(
        f"  {label:<30}  faces={len(filt_vecs):>2}/{len(frames)}"
        f"  norm(final)={np.linalg.norm(final):.6f}"
    )
    return final


# =============================================================================
#  COSINE SIMILARITY
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# =============================================================================
#  PATH VALIDATION
# =============================================================================

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
        f"Files in directory:\n" +
        "\n".join(f"  {p.name}" for p in sorted(video_dir.iterdir()))
    )


# =============================================================================
#  MAIN
# =============================================================================

SEP  = "=" * 72
SEP2 = "-" * 72


def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    video_entries = [
        (person, variant, resolve_video_path(VIDEO_DIR, filename))
        for person, variant, filename in VIDEOS
    ]

    detector = _load_retinaface() or _load_mtcnn()
    model    = AdaFaceModel(WEIGHTS_PATH)
    aligner  = FaceAligner(detector=detector)

    print(f"\n{SEP}")
    print("  HIGH-SIMILARITY PIPELINE  —  Bilateral Filter Edition")
    print(SEP)
    print(f"  Video directory  : {VIDEO_DIR}")
    print(f"  Step 1  Detector : {detector[0] if detector else 'Haar only'}")
    print( "  Step 1  IOD Check: eye-distance consistency filter (tol=20%)")
    print(f"  Step 1b Bilateral: d={BILATERAL_D}, sigmaColor={BILATERAL_SIGMA_COLOR}, sigmaSpace={BILATERAL_SIGMA_SPACE}")
    print( "  Step 2  Colour   : Gamma + LAB CLAHE + A/B neutral shift")
    print(f"  Step 3  Cluster  : K={K_SUBCENTERS} sub-centers (KMeans)")
    print( "  Step 4  Model    : AdaFace IR-18  (embed + H-flip)")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    labels:     List[str]             = []

    for person, variant, vpath in video_entries:
        key = f"{person}_{variant}"
        emb = embed_video(vpath, model, aligner,
                          f"{person} {variant.replace('_', ' ')}")
        if emb is None:
            raise RuntimeError(f"Embedding failed for {key}")
        embeddings[key] = emb
        labels.append(key)

    n = len(labels)

    def short(key: str) -> str:
        p, v = key.rsplit("_", 1)
        return f"{p}-{'MU' if v == 'makeup' else 'NM'}"

    slabels = [short(k) for k in labels]

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_sim(embeddings[labels[i]], embeddings[labels[j]])

    # ── SECTION 1: Within-identity pairs ──────────────────────────────────
    persons       = ["V8", "V9", "V10", "V11", "V12", "V13"]
    within_scores = []

    print(f"\n{SEP}")
    print("  WITHIN-IDENTITY  —  same person: makeup vs no-makeup")
    print(SEP)
    print(f"  {'Pair':<42}  {'Cosine Sim':>10}")
    print(f"  {'-'*42}  {'-'*10}")

    for person in persons:
        mu_emb = embeddings[f"{person}_makeup"]
        nm_emb = embeddings[f"{person}_no_makeup"]
        s      = cosine_sim(mu_emb, nm_emb)
        within_scores.append(s)
        print(f"  {person+' makeup':<20}  vs  {person+' no makeup':<18}  {s:>10.4f}")

    print()
    print(f"  Mean   : {np.mean(within_scores):.4f}")
    print(f"  Min    : {np.min(within_scores):.4f}  ({persons[int(np.argmin(within_scores))]})")
    print(f"  Max    : {np.max(within_scores):.4f}  ({persons[int(np.argmax(within_scores))]})")
    passing = sum(1 for s in within_scores if s >= THRESHOLD)
    print(f"  Passing: {passing}/{len(within_scores)} above {THRESHOLD}")

    # ── SECTION 2: Full 12x12 matrix ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  FULL 12x12 COSINE SIMILARITY MATRIX")
    print(SEP)
    print("  MU = Makeup   NM = No makeup   Diagonal = 1.000 (self)")
    print()

    col_w  = 8
    header = " " * 10 + "".join(f"{s:>{col_w}}" for s in slabels)
    print(f"  {header}")
    print(f"  {'-'*(10 + col_w*n)}")
    for i, row_lbl in enumerate(slabels):
        row = f"  {row_lbl:<8}"
        for j in range(n):
            row += f"{sim[i,j]:>{col_w}.4f}"
        print(row)

    # ── SECTION 3: All pairwise sorted ────────────────────────────────────
    print(f"\n{SEP}")
    print("  ALL PAIRWISE COSINE SIMILARITIES  (sorted high -> low)")
    print(SEP)
    print(f"  {'Pair':<45}  {'Cosine Sim':>10}  Type")
    print(f"  {'-'*45}  {'-'*10}  {'-'*16}")

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pi    = labels[i].split("_")[0]
            pj    = labels[j].split("_")[0]
            ptype = "same person" if pi == pj else "different person"
            pairs.append((slabels[i], slabels[j], sim[i, j], ptype))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for la, lb, s, pt in pairs:
        print(f"  {la:<20}  vs  {lb:<22}  {s:>10.4f}  {pt}")

    # ── SECTION 4: Summary ────────────────────────────────────────────────
    cross_scores = [s for _, _, s, pt in pairs if pt == "different person"]

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Step 1   Detector   : {detector[0] if detector else 'Haar only'}")
    print( "  Step 1   IOD filter : eye-distance consistency (tol=20%)")
    print(f"  Step 1b  Bilateral  : d={BILATERAL_D}, sigmaColor={BILATERAL_SIGMA_COLOR}, sigmaSpace={BILATERAL_SIGMA_SPACE}")
    print( "  Step 2   Colour     : Gamma + LAB CLAHE + A/B neutral")
    print(f"  Step 3   Cluster    : K={K_SUBCENTERS} sub-centers")
    print( "  Step 4   Model      : AdaFace IR-18")
    print()
    print(f"  Within-identity mean    : {np.mean(within_scores):.4f}")
    print(f"  Cross-identity  mean    : {np.mean(cross_scores):.4f}")
    gap = np.mean(within_scores) - np.mean(cross_scores)
    print(f"  Separation gap          : {gap:.4f}  "
          f"({'Good' if gap > 0.15 else 'Low'} separation)")
    print(f"  Threshold               : {THRESHOLD}")
    print(f"  Within-identity passing : {passing}/{len(within_scores)}")
    print()
    print("  Per-person within-identity similarity:")
    for person, s in zip(persons, within_scores):
        print(f"    {person:<5}  {s:.4f}")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
