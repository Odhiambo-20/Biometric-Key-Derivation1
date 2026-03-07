"""
AdaFace 5-Phase Pipeline
=========================

Phase 1 : Build Master V
          V1+V2+V3+V4 → average → L2 renorm → Master V (single enrolled template)

Phase 2 : Quantise V1, V2, V3, V4, Master V → 4-bit integers

Phase 3 : BER Tests (genuine)
          Test 1 — V1,V2,V3,V4 against each other
          Test 2 — V1,V2,V3,V4 each against Master V

Phase 4 : Quantise V5, V6, V7 → 4-bit integers

Phase 5 : FAR Test (impostor)
          V5, V6, V7 each against Master V

Embedding pipeline per video:
  1. Extract top-20 sharpest frames (scan 60 candidates)
  2. Detect face + eye cascade → Umeyama 112×112 alignment
  3. AdaFace IR-18 → raw 512-dim embedding
  4. L2 normalise → unit vector
  5. Average all frame unit vectors → L2 renormalise → final unit embedding
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
IMPOSTOR_SET   = {"video_5", "video_6", "video_7"}
FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112
QUANT_BITS     = 4          # 4-bit quantisation → values 0..15

# AdaFace canonical 5-point reference landmarks (112×112)
REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — FACE ALIGNER
# ─────────────────────────────────────────────────────────────────────────────

class FaceAligner:
    def __init__(self):
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        if self.face_cascade.empty():
            raise RuntimeError("Face Haar cascade not found.")

        eye_xml       = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.eye_ok   = not self.eye_cascade.empty()
        self.clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        log.info(f"FaceAligner ready  |  eye cascade: {'yes' if self.eye_ok else 'no (geometry fallback)'}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _detect_face(self, gray: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        for sf, mn, ms in [(1.05, 6, 60), (1.05, 3, 40), (1.10, 2, 30)]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms)
            )
            if len(faces) > 0:
                return tuple(max(faces, key=lambda f: f[2] * f[3]))
        return None

    def _detect_eyes(
        self, gray: np.ndarray, fx: int, fy: int, fw: int, fh: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.eye_ok:
            return None
        roi = gray[fy : fy + int(fh * 0.60), fx : fx + fw]
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20)
        )
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.10, minNeighbors=2, minSize=(15, 15)
            )
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        centres = sorted(
            [np.array([fx + ex + ew//2, fy + ey + eh//2], dtype=np.float32)
             for ex, ey, ew, eh in eyes],
            key=lambda p: p[0]
        )
        return centres[0], centres[1]

    @staticmethod
    def _landmarks(x, y, w, h, le=None, re=None) -> np.ndarray:
        le = le if le is not None else np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        re = re if re is not None else np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        return np.array([
            le,
            re,
            [x+0.50*w, y+0.57*h],
            [x+0.35*w, y+0.76*h],
            [x+0.65*w, y+0.76*h],
        ], dtype=np.float32)

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        n = src.shape[0]
        mu_s = src.mean(0); mu_d = dst.mean(0)
        sc = src - mu_s;    dc = dst - mu_d
        vs = (sc**2).sum() / n
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
        M[:, :2] = c * R; M[:, 2] = t
        return M

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = self._preprocess(frame)
        det  = self._detect_face(gray)
        if det is None:
            return None
        x, y, w, h = det
        eyes   = self._detect_eyes(gray, x, y, w, h)
        le, re = (eyes[0], eyes[1]) if eyes else (None, None)
        src    = self._landmarks(x, y, w, h, le, re)
        M      = self._umeyama(src, REFERENCE_PTS)
        if M is None:
            fh, fw = frame.shape[:2]
            crop = frame[max(0,y-int(h*.05)):min(fh,y+h+int(h*.02)),
                         max(0,x-int(w*.10)):min(fw,x+w+int(w*.10))]
            return cv2.resize(crop, (FACE_SIZE, FACE_SIZE),
                              interpolation=cv2.INTER_LANCZOS4) if crop.size else None
        return cv2.warpAffine(frame, M, (FACE_SIZE, FACE_SIZE),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — ADAFACE MODEL
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
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan    = FRAMES_TO_USE * CANDIDATE_MULT
    positions = [int(round(i*(total-1)/max(n_scan-1,1))) for i in range(n_scan)]
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
        raise RuntimeError(f"No frames: {video_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:FRAMES_TO_USE]
    top.sort(key=lambda x: x[1])
    return [f for _, _, f in top]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — EMBED ONE VIDEO → UNIT EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_video(
    video_path : str,
    model      : AdaFaceModel,
    aligner    : FaceAligner,
) -> Optional[np.ndarray]:
    frames    = extract_frames(video_path)
    unit_vecs = []
    for frame in frames:
        aligned = aligner.align(frame)
        if aligned is None:
            continue
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
        return None
    final = (avg / avg_norm).astype(np.float32)
    log.info(f"  {Path(video_path).name:<44}  faces={len(unit_vecs):>2}/{len(frames)}  norm={np.linalg.norm(final):.6f}")
    return final


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — QUANTISATION  (4-bit)
#
#  AdaFace embeddings are L2-normalised → each dimension ∈ [-1, +1].
#  4-bit uniform quantisation:
#    step  = 2.0 / (2^4 - 1) = 2.0 / 15 ≈ 0.1333
#    q(v)  = round((v + 1.0) / step)  clamped to [0, 15]
#
#  This maps:
#    v = -1.0  →  q = 0
#    v =  0.0  →  q = 7 or 8
#    v = +1.0  →  q = 15
#
#  BER is computed bit-by-bit over the 4-bit binary representation
#  of each quantised value. Total bits = 512 dims × 4 bits = 2048 bits.
# ─────────────────────────────────────────────────────────────────────────────

def quantise(embedding: np.ndarray, bits: int = QUANT_BITS) -> np.ndarray:
    """
    FP32 unit embedding → integer array of shape (512,), values in [0, 2^bits-1].
    Uniform quantisation over [-1, +1].
    """
    levels = (1 << bits) - 1          # 15 for 4-bit
    q = np.round((embedding + 1.0) / 2.0 * levels).astype(np.int32)
    return np.clip(q, 0, levels)


def to_bits(q_vec: np.ndarray, bits: int = QUANT_BITS) -> np.ndarray:
    """
    Integer array (512,) → binary bit array (512 × bits,) = (2048,).
    Each integer is unpacked MSB-first into `bits` bits.
    """
    n      = len(q_vec)
    result = np.zeros(n * bits, dtype=np.uint8)
    for i, val in enumerate(q_vec):
        for b in range(bits):
            result[i * bits + (bits - 1 - b)] = (int(val) >> b) & 1
    return result


def ber(bits_a: np.ndarray, bits_b: np.ndarray) -> Tuple[int, float]:
    """
    Bit Error Rate between two binary arrays.
    Returns (num_errors, error_rate_percent).
    """
    errors = int(np.sum(bits_a != bits_b))
    rate   = errors / len(bits_a) * 100.0
    return errors, rate


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


SEP  = "=" * 62
SEP2 = "─" * 62


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

    # ── Extract embeddings for all 7 videos ──────────────────────────────────
    print(f"\n{SEP}")
    print("  EXTRACTING EMBEDDINGS — ALL 7 VIDEOS")
    print(SEP)
    print(f"  {FRAMES_TO_USE} frames | CLAHE | Eye cascade | Umeyama 112×112")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        key    = f"video_{idx}"
        result = embed_video(vp, model, aligner)
        if result is not None:
            embeddings[key] = result

    # ── Cosine similarity table (all pairs) ──────────────────────────────────
    print(f"\n{SEP}")
    print("  COSINE SIMILARITY — ALL PAIRS  (reference check)")
    print(SEP)
    print(f"\n  {'Pair':<35}  {'Similarity':>10}")
    print(f"  {'─'*35}  {'─'*10}")

    all_keys = [f"video_{i}" for i in range(1, 8) if f"video_{i}" in embeddings]
    for i, ka in enumerate(all_keys):
        for kb in all_keys[i+1:]:
            sim = cosine_sim(embeddings[ka], embeddings[kb])
            na  = VIDEO_NAMES.get(ka, ka)
            nb  = VIDEO_NAMES.get(kb, kb)
            print(f"  {na + ' vs ' + nb:<35}  {sim:>10.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — BUILD MASTER V
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PHASE 1 — BUILD MASTER V")
    print(SEP)
    print("  Average V1+V2+V3+V4 unit embeddings → L2 renorm → Master V")
    print()

    genuine_keys = [k for k in ["video_1","video_2","video_3","video_4"]
                    if k in embeddings]
    if len(genuine_keys) < 4:
        raise RuntimeError("Not all genuine videos (V1–V4) were embedded successfully.")

    genuine_stack = np.stack([embeddings[k] for k in genuine_keys], axis=0)
    master_avg    = np.mean(genuine_stack, axis=0).astype(np.float32)
    master_norm   = np.linalg.norm(master_avg)
    master_v      = (master_avg / master_norm).astype(np.float32)

    print(f"  Inputs  : {', '.join(VIDEO_NAMES[k] for k in genuine_keys)}")
    print(f"  Average norm (before renorm) : {master_norm:.6f}")
    print(f"  Master V norm (after renorm) : {np.linalg.norm(master_v):.8f}")
    print()

    # Cosine similarity of each genuine video to Master V
    print(f"  Cosine similarity to Master V:")
    for k in genuine_keys:
        sim = cosine_sim(embeddings[k], master_v)
        print(f"    {VIDEO_NAMES[k]:<20}  →  {sim:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — QUANTISE V1, V2, V3, V4, MASTER V  (4-bit)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PHASE 2 — QUANTISE GENUINE EMBEDDINGS + MASTER V  (4-bit)")
    print(SEP)
    print(f"  Method  : uniform quantisation over [-1, +1]")
    print(f"  Bits    : {QUANT_BITS}  → levels 0..{(1<<QUANT_BITS)-1}")
    print(f"  Dims    : 512")
    print(f"  Total bits per embedding : {512 * QUANT_BITS}")
    print()

    # Quantise genuine videos
    q_genuine: Dict[str, np.ndarray] = {}   # integer arrays
    b_genuine: Dict[str, np.ndarray] = {}   # bit arrays

    for k in genuine_keys:
        q = quantise(embeddings[k])
        b = to_bits(q)
        q_genuine[k] = q
        b_genuine[k] = b
        name = VIDEO_NAMES[k]
        print(f"  {name:<20}  q[:8]={q[:8].tolist()}  bits[:16]={b[:16].tolist()}")

    # Quantise Master V
    q_master = quantise(master_v)
    b_master = to_bits(q_master)
    print(f"  {'Master V':<20}  q[:8]={q_master[:8].tolist()}  bits[:16]={b_master[:16].tolist()}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3 — BER TESTS  (genuine)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PHASE 3 — BER TESTS  (GENUINE)")
    print(SEP)
    print(f"  BER = Bit Error Rate over {512*QUANT_BITS} bits ({512} dims × {QUANT_BITS} bits)")
    print(f"  Expected: LOW BER for same-person pairs")

    # ── Test 1: V1,V2,V3,V4 vs each other ────────────────────────────────────
    print(f"\n  TEST 1 — Genuine pairs  (V1,V2,V3,V4 vs each other)")
    print(f"  {SEP2}")
    print(f"  {'Pair':<35}  {'Errors':>7}  {'BER %':>7}  {'CosSim':>8}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*7}  {'─'*8}")

    t1_bers = []
    for i, ka in enumerate(genuine_keys):
        for kb in genuine_keys[i+1:]:
            errs, rate = ber(b_genuine[ka], b_genuine[kb])
            sim        = cosine_sim(embeddings[ka], embeddings[kb])
            na         = VIDEO_NAMES[ka]
            nb         = VIDEO_NAMES[kb]
            t1_bers.append(rate)
            print(f"  {na + ' vs ' + nb:<35}  {errs:>7}  {rate:>6.2f}%  {sim:>8.4f}")

    print(f"\n  Genuine vs Genuine  —  "
          f"min BER={min(t1_bers):.2f}%  "
          f"max BER={max(t1_bers):.2f}%  "
          f"mean BER={np.mean(t1_bers):.2f}%")

    # ── Test 2: V1,V2,V3,V4 each vs Master V ─────────────────────────────────
    print(f"\n  TEST 2 — Genuine videos vs Master V")
    print(f"  {SEP2}")
    print(f"  {'Pair':<35}  {'Errors':>7}  {'BER %':>7}  {'CosSim':>8}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*7}  {'─'*8}")

    t2_bers = []
    for k in genuine_keys:
        errs, rate = ber(b_genuine[k], b_master)
        sim        = cosine_sim(embeddings[k], master_v)
        name       = VIDEO_NAMES[k]
        t2_bers.append(rate)
        print(f"  {name + ' vs Master V':<35}  {errs:>7}  {rate:>6.2f}%  {sim:>8.4f}")

    print(f"\n  Genuine vs Master V —  "
          f"min BER={min(t2_bers):.2f}%  "
          f"max BER={max(t2_bers):.2f}%  "
          f"mean BER={np.mean(t2_bers):.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 4 — QUANTISE V5, V6, V7  (4-bit)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PHASE 4 — QUANTISE IMPOSTOR EMBEDDINGS  (4-bit)")
    print(SEP)
    print(f"  Same quantisation scheme: uniform [-1,+1] → {QUANT_BITS}-bit integers")
    print()

    impostor_keys = [k for k in ["video_5","video_6","video_7"]
                     if k in embeddings]

    q_impostor: Dict[str, np.ndarray] = {}
    b_impostor: Dict[str, np.ndarray] = {}

    for k in impostor_keys:
        q = quantise(embeddings[k])
        b = to_bits(q)
        q_impostor[k] = q
        b_impostor[k] = b
        name = VIDEO_NAMES[k]
        print(f"  {name:<20}  q[:8]={q[:8].tolist()}  bits[:16]={b[:16].tolist()}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 5 — FAR TEST  (V5,V6,V7 vs Master V)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PHASE 5 — FAR TEST  (IMPOSTOR vs MASTER V)")
    print(SEP)
    print(f"  Expected: HIGH BER — different people → near-random bits")
    print()
    print(f"  {'Pair':<35}  {'Errors':>7}  {'BER %':>7}  {'CosSim':>8}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*7}  {'─'*8}")

    t5_bers = []
    for k in impostor_keys:
        errs, rate = ber(b_impostor[k], b_master)
        sim        = cosine_sim(embeddings[k], master_v)
        name       = VIDEO_NAMES[k]
        t5_bers.append(rate)
        print(f"  {name + ' vs Master V':<35}  {errs:>7}  {rate:>6.2f}%  {sim:>8.4f}")

    print(f"\n  Impostor vs Master V — "
          f"min BER={min(t5_bers):.2f}%  "
          f"max BER={max(t5_bers):.2f}%  "
          f"mean BER={np.mean(t5_bers):.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(SEP)
    print(f"  Quantisation          : {QUANT_BITS}-bit uniform  |  {512*QUANT_BITS} bits per embedding")
    print()
    print(f"  Phase 3 Test 1  Genuine vs Genuine  :  "
          f"BER {min(t1_bers):.2f}% – {max(t1_bers):.2f}%  "
          f"(mean {np.mean(t1_bers):.2f}%)")
    print(f"  Phase 3 Test 2  Genuine vs Master V :  "
          f"BER {min(t2_bers):.2f}% – {max(t2_bers):.2f}%  "
          f"(mean {np.mean(t2_bers):.2f}%)")
    print(f"  Phase 5         Impostor vs Master V:  "
          f"BER {min(t5_bers):.2f}% – {max(t5_bers):.2f}%  "
          f"(mean {np.mean(t5_bers):.2f}%)")
    print()

    all_genuine_bers  = t1_bers + t2_bers
    genuine_max_ber   = max(all_genuine_bers)
    impostor_min_ber  = min(t5_bers)
    gap               = impostor_min_ber - genuine_max_ber

    print(f"  Genuine  max BER  : {genuine_max_ber:.2f}%")
    print(f"  Impostor min BER  : {impostor_min_ber:.2f}%")
    print(f"  Separation gap    : {gap:+.2f}%  "
          f"{'CLEAN ✓' if gap > 0 else 'OVERLAP ✗'}")

    if gap > 0:
        bch_t_percent = (genuine_max_ber + impostor_min_ber) / 2.0
        print()
        print(f"  BCH threshold t should correct up to ~{bch_t_percent:.1f}% BER")
        print(f"  → sits midway between genuine max ({genuine_max_ber:.2f}%)"
              f" and impostor min ({impostor_min_ber:.2f}%)")

    print(SEP)


if __name__ == "__main__":
    main()
