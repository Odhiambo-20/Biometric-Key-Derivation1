"""
AdaFace Cosine Similarity Pipeline
====================================
Videos: V8–V13, each with Make up / No make up variant.

Pipeline per video:
  1.  Extract top-20 sharpest frames (scan 60 candidates)
  2.  Detect face + eye cascade → Umeyama 112×112 alignment
  3.  AdaFace IR-18 → raw 512-dim embedding
  4.  L2-normalise each frame vector → unit vector
  5.  Average all frame unit vectors
  6.  L2-renormalise → final unit vector
  7.  Pairwise cosine similarity across all 12 videos
      (cosine similarity = dot product of two unit vectors)

Output:
  - 12×12 cosine similarity matrix
  - Within-identity pairs (MakeUp vs NoMakeUp for same person)
  - Cross-identity pairs (all other combinations)
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

# =============================================================================
#  CONFIG
# =============================================================================

WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)

VIDEOS = [
    ("V8",  "Make up",    "/home/victor/Documents/Desktop/Embeddings/V8 instagram Make up.mp4"),
    ("V8",  "No make up", "/home/victor/Documents/Desktop/Embeddings/V8 instagram No make up.mp4"),
    ("V9",  "Make up",    "/home/victor/Documents/Desktop/Embeddings/V9 Instagram Make up.mp4"),
    ("V9",  "No make up", "/home/victor/Documents/Desktop/Embeddings/V9 Instagram No make up.mp4"),
    ("V10", "Make up",    "/home/victor/Documents/Desktop/Embeddings/V10 Instagram Make up.mp4"),
    ("V10", "No make up", "/home/victor/Documents/Desktop/Embeddings/V10 Instagram No make up.mp4"),
    ("V11", "Make up",    "/home/victor/Documents/Desktop/Embeddings/V11 instagram Make up.mp4"),
    ("V11", "No make up", "/home/victor/Documents/Desktop/Embeddings/V11 instagram No make up.mp4"),
    ("V12", "Make up",    "/home/victor/Documents/Desktop/Embeddings/V12 Instagram Make up.mp4"),
    ("V12", "No make up", "/home/victor/Documents/Desktop/Embeddings/V12 Instagram No make up.mp4"),
    ("V13", "Make up",    "/home/victor/Documents/Desktop/Embeddings/V13 instagram Make up.mp4"),
    ("V13", "No make up", "/home/victor/Documents/Desktop/Embeddings/V13 Instagram No make up.mp4"),
]

FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3      # scan 60 candidates, keep top-20 by sharpness
FACE_SIZE      = 112

REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# =============================================================================
#  FACE ALIGNER
# =============================================================================

class FaceAligner:
    def __init__(self):
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        if self.face_cascade.empty():
            raise RuntimeError("Face Haar cascade not found.")
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade  = cv2.CascadeClassifier(eye_xml)
        self.eye_ok       = not self.eye_cascade.empty()
        self.clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        log.info(
            "FaceAligner | eye cascade: "
            + ("yes" if self.eye_ok else "no (geometry fallback)")
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        return self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def _detect_face(self, gray: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        for sf, mn, ms in [(1.05, 6, 60), (1.05, 3, 40), (1.10, 2, 30)]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
            if len(faces) > 0:
                return tuple(max(faces, key=lambda f: f[2]*f[3]))
        return None

    def _detect_eyes(self, gray, fx, fy, fw, fh):
        if not self.eye_ok:
            return None
        roi  = gray[fy : fy + int(fh * 0.60), fx : fx + fw]
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20))
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.10, minNeighbors=2, minSize=(15, 15))
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        centres = sorted(
            [np.array([fx+ex+ew//2, fy+ey+eh//2], dtype=np.float32)
             for ex, ey, ew, eh in eyes],
            key=lambda p: p[0],
        )
        return centres[0], centres[1]

    @staticmethod
    def _landmarks(x, y, w, h, le=None, re=None) -> np.ndarray:
        le = le if le is not None else np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        re = re if re is not None else np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        return np.array([
            le, re,
            [x+0.50*w, y+0.57*h],
            [x+0.35*w, y+0.76*h],
            [x+0.65*w, y+0.76*h],
        ], dtype=np.float32)

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        n    = src.shape[0]
        mu_s = src.mean(0); mu_d = dst.mean(0)
        sc   = src - mu_s;  dc   = dst - mu_d
        vs   = (sc**2).sum() / n
        if vs < 1e-10:
            return None
        cov = (dc.T @ sc) / n
        try:
            U, S, Vt = np.linalg.svd(cov)
        except np.linalg.LinAlgError:
            return None
        d  = np.ones(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            d[-1] = -1
        R  = U @ np.diag(d) @ Vt
        c  = (S * d).sum() / vs
        t  = mu_d - c * R @ mu_s
        M  = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = c * R
        M[:,  2] = t
        return M

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = self._preprocess(frame)
        det  = self._detect_face(gray)
        if det is None:
            return None
        x, y, w, h = det
        eyes = self._detect_eyes(gray, x, y, w, h)
        le, re = (eyes[0], eyes[1]) if eyes else (None, None)
        src  = self._landmarks(x, y, w, h, le, re)
        M    = self._umeyama(src, REFERENCE_PTS)
        if M is None:
            fh, fw = frame.shape[:2]
            crop = frame[
                max(0, y-int(h*0.05)) : min(fh, y+h+int(h*0.02)),
                max(0, x-int(w*0.10)) : min(fw, x+w+int(w*0.10)),
            ]
            if crop.size == 0:
                return None
            return cv2.resize(crop, (FACE_SIZE, FACE_SIZE),
                              interpolation=cv2.INTER_LANCZOS4)
        return cv2.warpAffine(frame, M, (FACE_SIZE, FACE_SIZE),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)


# =============================================================================
#  ADAFACE MODEL
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

    def raw_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        return emb.astype(np.float32)


# =============================================================================
#  FRAME EXTRACTION  (top-20 sharpest out of 60 candidates)
# =============================================================================

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan   = FRAMES_TO_USE * CANDIDATE_MULT
    positions = [
        int(round(i * (total - 1) / max(n_scan - 1, 1)))
        for i in range(n_scan)
    ]
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
        raise RuntimeError(f"No frames found: {video_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:FRAMES_TO_USE]
    top.sort(key=lambda x: x[1])
    return [f for _, _, f in top]


# =============================================================================
#  EMBED VIDEO
#  Steps: raw → L2-normalise per frame → average → L2-renormalise
# =============================================================================

def embed_video(
    video_path: str,
    model: AdaFaceModel,
    aligner: FaceAligner,
    label: str,
) -> Optional[np.ndarray]:
    frames     = extract_frames(video_path)
    unit_vecs  = []

    for frame in frames:
        aligned = aligner.align(frame)
        if aligned is None:
            continue
        raw  = model.raw_embedding(aligned)
        # Step 1: L2-normalise each frame embedding
        norm = np.linalg.norm(raw)
        if norm < 1e-10:
            continue
        unit_vecs.append((raw / norm).astype(np.float32))

    if not unit_vecs:
        log.error(f"No faces detected: {Path(video_path).name}")
        return None

    # Step 2: Average the unit vectors
    avg = np.mean(np.stack(unit_vecs, axis=0), axis=0).astype(np.float32)

    # Step 3: L2-renormalise the average
    avg_norm = np.linalg.norm(avg)
    if avg_norm < 1e-10:
        return None
    final = (avg / avg_norm).astype(np.float32)

    log.info(
        f"  {label:<30}  faces={len(unit_vecs):>2}/{len(frames)}"
        f"  norm={np.linalg.norm(final):.6f}"
    )
    return final


# =============================================================================
#  COSINE SIMILARITY  (dot product of unit vectors = cosine of angle)
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# =============================================================================
#  MAIN
# =============================================================================

SEP  = "=" * 80
SEP2 = "-" * 80

def main() -> None:
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for _, _, vp in VIDEOS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model   = AdaFaceModel(WEIGHTS_PATH)
    aligner = FaceAligner()

    # ── Extract all embeddings ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EMBEDDING EXTRACTION  V8–V13  (Make up / No make up)")
    print(SEP)
    print(f"  Pipeline per video:")
    print(f"    1. Extract top-{FRAMES_TO_USE} sharpest frames from {FRAMES_TO_USE*CANDIDATE_MULT} candidates")
    print(f"    2. Face detect → eye cascade → Umeyama align → {FACE_SIZE}×{FACE_SIZE}")
    print(f"    3. AdaFace IR-18 → 512-dim raw embedding")
    print(f"    4. L2-normalise each frame vector")
    print(f"    5. Average normalised vectors")
    print(f"    6. L2-renormalise → final unit vector")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    labels:     List[str]             = []

    for person, variant, vpath in VIDEOS:
        label = f"{person} {variant}"
        emb   = embed_video(vpath, model, aligner, label)
        if emb is None:
            raise RuntimeError(f"Embedding failed for {label}")
        key = label
        embeddings[key] = emb
        labels.append(key)

    n = len(labels)

    # ── Build similarity matrix ────────────────────────────────────────────
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_similarity(embeddings[labels[i]], embeddings[labels[j]])

    # ── Print full matrix ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COSINE SIMILARITY MATRIX  (all 12 videos × 12 videos)")
    print(SEP)
    print(f"  Diagonal = 1.000 (self-similarity)")
    print(f"  Higher value = more similar face identity")
    print()

    # Short labels for matrix display
    short = [
        f"V{lbl.split()[0][1:]}-{'MU' if 'Make' in lbl else 'NM'}"
        for lbl in labels
    ]

    col_w = 8
    header = " " * 12 + "".join(f"{s:>{col_w}}" for s in short)
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for i, row_lbl in enumerate(short):
        row = f"  {row_lbl:<10}"
        for j in range(n):
            val = sim[i, j]
            row += f"{val:>{col_w}.4f}"
        print(row)

    # ── Within-identity pairs (same person, MakeUp vs NoMakeUp) ───────────
    print(f"\n{SEP}")
    print("  WITHIN-IDENTITY PAIRS  (same person — Make up vs No make up)")
    print(SEP)
    print(f"  {'Pair':<42}  {'Cosine Sim':>10}  {'Interpretation'}")
    print(f"  {'-'*42}  {'-'*10}  {'-'*30}")

    within_sims = []
    persons = ["V8", "V9", "V10", "V11", "V12", "V13"]
    for person in persons:
        mu_key = f"{person} Make up"
        nm_key = f"{person} No make up"
        s = cosine_similarity(embeddings[mu_key], embeddings[nm_key])
        within_sims.append(s)
        interp = (
            "Very strong match" if s >= 0.50 else
            "Strong match"      if s >= 0.35 else
            "Moderate match"    if s >= 0.20 else
            "Weak match"        if s >= 0.10 else
            "Poor match"
        )
        print(f"  {mu_key:<20}  vs  {nm_key:<18}  {s:>10.4f}  {interp}")

    print(f"\n  Mean within-identity cosine similarity : {np.mean(within_sims):.4f}")
    print(f"  Min  within-identity cosine similarity : {np.min(within_sims):.4f}")
    print(f"  Max  within-identity cosine similarity : {np.max(within_sims):.4f}")

    # ── Cross-identity pairs (different persons) ───────────────────────────
    print(f"\n{SEP}")
    print("  CROSS-IDENTITY PAIRS  (different persons)")
    print(SEP)
    print(f"  {'Pair':<45}  {'Cosine Sim':>10}")
    print(f"  {'-'*45}  {'-'*10}")

    cross_sims = []
    cross_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pi = labels[i].split()[0]  # e.g. "V8"
            pj = labels[j].split()[0]
            if pi != pj:
                s = sim[i, j]
                cross_sims.append(s)
                cross_pairs.append((labels[i], labels[j], s))

    cross_pairs.sort(key=lambda x: x[2], reverse=True)
    for la, lb, s in cross_pairs:
        print(f"  {la:<22}  vs  {lb:<20}  {s:>10.4f}")

    print(f"\n  Mean cross-identity cosine similarity : {np.mean(cross_sims):.4f}")
    print(f"  Min  cross-identity cosine similarity : {np.min(cross_sims):.4f}")
    print(f"  Max  cross-identity cosine similarity : {np.max(cross_sims):.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Within-identity mean  : {np.mean(within_sims):.4f}")
    print(f"  Cross-identity  mean  : {np.mean(cross_sims):.4f}")
    gap = np.mean(within_sims) - np.mean(cross_sims)
    print(f"  Separation gap        : {gap:.4f}  "
          f"({'Good separation' if gap > 0.10 else 'Low separation — makeup causes high intra-person variation'})")
    print()
    print("  Per-person within-identity similarity:")
    for person, s in zip(persons, within_sims):
        bar = "█" * int(s * 40)
        print(f"    {person:<4}  {s:.4f}  {bar}")
    print(SEP)


if __name__ == "__main__":
    main()
