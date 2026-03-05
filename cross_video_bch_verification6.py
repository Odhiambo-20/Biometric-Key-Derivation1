"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAFACE FACE EMBEDDING PIPELINE  +  BCH FUZZY COMMITMENT            ║
║                Phase 11  —  QUANT_BITS=4  +  V5 IMPOSTOR TEST             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BCH PARAMETERS  (Phase 11)                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  QUANT_BITS  : 4   →  payload = 512 × 4 = 2048 bits                        ║
║  Quant levels: 2^4 = 16 levels per dimension  (0 – 15)                     ║
║                                                                              ║
║  BCH(N=255, t=25)  →  PAR=184, K=71                                        ║
║    Chunks   = ceil(2048 / 71) = 29                                          ║
║    Padding  = 29 × 71 − 2048  = 11 bits  (last chunk zero-padded)          ║
║    t_total  = 29 × 25         = 725                                         ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  VIDEOS                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  V1  IOS Beard                ← ENROLLED (identity anchor)                  ║
║  V2  IOS No-Beard             ← same person, different condition → PASS     ║
║  V3  Android Beard            ← same person, different device    → PASS     ║
║  V4  Android No-Beard         ← same person, both differ         → PASS     ║
║  V5  Android video 5          ← DIFFERENT person (impostor)      → FAIL    ║
║                                                                              ║
║  V5 uses the SAME V1 helper data and hash key.  A different person's        ║
║  embedding will produce bit differences far beyond t in most chunks,        ║
║  so BCH decoding will fail and the recovered hash will NOT match H1.        ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  PIPELINE OVERVIEW                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1.  Extract top-20 sharpest frames from each video (Laplacian variance)    ║
║  2.  Detect face with Haar cascade → tight crop                             ║
║  3.  Resize to 112×112, apply lower-face mask (bottom 38 %)                ║
║  4.  AdaFace IR-18 ONNX → 512-dim embedding, L2-normalised per frame       ║
║  5.  Average all per-frame embeddings → re-normalise → FINAL embedding      ║
║  6.  Pairwise cosine similarities across ALL 5 videos                       ║
║  7.  BCH Fuzzy-Commitment  (enroll V1, verify V2 / V3 / V4 / V5)           ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BCH FUZZY-COMMITMENT SCHEME  (Juels & Wattenberg, 1999)                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  ENROLL (V1):                                                                ║
║    Quantise V1 → 2048-bit payload  (4 bits × 512 dims)                     ║
║    Split into 29 chunks of K=71 bits  (last chunk zero-padded 11 bits)     ║
║    For each chunk i:                                                         ║
║      r[i]       = V1_chunk[i]                   ← 71-bit secret             ║
║      c_r[i]     = BCH_encode(r[i])              ← 255-bit codeword          ║
║      v1_pad[i]  = r[i] ++ zeros(184)            ← extend r to 255 bits      ║
║      helper[i]  = c_r[i] XOR v1_pad[i]          ← stored helper data        ║
║    hash_key = SHA-256( r[0] ‖ r[1] ‖ … ‖ r[28] )  ← stored hash           ║
║                                                                              ║
║  VERIFY (Vx):                                                                ║
║    Quantise Vx using V1's [v_min, v_max]  →  2048-bit payload              ║
║    For each chunk i:                                                         ║
║      vx_pad[i]  = Vx_chunk[i] ++ zeros(184)     ← extend to 255 bits       ║
║      noisy[i]   = helper[i] XOR vx_pad[i]                                  ║
║                 = c_r[i] XOR (r[i]⊕Vx_chunk[i]) ++ zeros(184)             ║
║                 = c_r[i] with errors at positions where Vx ≠ V1            ║
║      r̂[i]      = BCH_decode(noisy[i])           ← recovers r[i] if ≤ t    ║
║    hash_verify = SHA-256( r̂[0] ‖ … ‖ r̂[28] )                            ║
║    PASS  iff  hash_verify == hash_key                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

MASK:  bottom 38 % blacked out  →  rows 70–111  (mouth + chin hidden)
       visible rows 0–69        →  forehead, eyebrows, eyes, nose, nostrils
"""

import hashlib
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# V1–V4: same person (enrolled identity).  V5: different person (impostor).
VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",                    # V1
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",        # V2
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",               # V3
    "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4",    # V4
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",        # V5 impostor
]

# Human-readable labels — must stay aligned with VIDEO_PATHS order.
VIDEO_LABELS = [
    "IOS Beard          (V1)",
    "IOS No-Beard       (V2)",
    "Android Beard      (V3)",
    "Android No-Beard   (V4)",
    "Android Video 5    (V5)",   # impostor
]

# Indices into VIDEO_PATHS / VIDEO_LABELS — change here to re-configure.
ENROLL_INDEX   = 0   # V1 is the enrolled identity
IMPOSTOR_INDEX = 4   # V5 is the impostor

WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)
OUTPUT_ROOT = "masked_frames"

# Frame extraction
FRAMES_TO_USE        = 20   # top-N sharpest kept
CANDIDATE_MULTIPLIER = 3    # scan pool = 20 × 3 = 60 candidates
FACE_SIZE            = 112  # AdaFace input (pixels)
MASK_FRACTION        = 0.38 # bottom 38 % blacked out

# Similarity
MIN_SIMILARITY_THRESHOLD = 0.80

# ── BCH  (Phase 11) ──────────────────────────────────────────────────────────
#
#  QUANT_BITS = 4  →  payload = 512 × 4 = 2048 bits
#  16 quantisation levels per dimension  (0 – 15)
#
#  BCH(N=255, t=25)  →  PAR=184, K=71
#    chunks  = ceil(2048 / 71) = 29
#    padding = 29×71 − 2048   = 11 bits  (last chunk zero-padded)
#    t_total = 29 × 25        = 725

BCH_N          = 255
BCH_T_DESIGNED = 25    # errors correctable per chunk
BCH_K          = 71    # message bits per chunk   (255 − 184)
BCH_PAR        = 184   # parity bits per chunk    (deg of generator)
QUANT_BITS     = 4     # 4-bit quantisation → 16 levels → 2048-bit payload
NUM_CHUNKS     = 29    # ceil(2048 / 71)
T_TOTAL        = 725   # 29 × 25
PAD_NEEDED     = NUM_CHUNKS * BCH_K - (512 * QUANT_BITS)   # 29×71 − 2048 = 11


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FACE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def apply_mask(image: np.ndarray) -> np.ndarray:
    img        = image.copy()
    black_rows = int(img.shape[0] * MASK_FRACTION)
    img[-black_rows:, :] = 0
    return img


def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(xml)
        if self.det.empty():
            raise RuntimeError("Haar cascade XML not found.")
        log.info("Face detector ready.")

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if w < 80 or h < 80:
            return None
        fh, fw = frame.shape[:2]
        x1 = max(0,  x - int(w * 0.10))
        y1 = max(0,  y - int(h * 0.05))
        x2 = min(fw, x + w + int(w * 0.10))
        y2 = min(fh, y + h + int(h * 0.02))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None


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
        log.info(f"Model loaded  |  Provider: {providers[0]}")

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]       # (1, 3, 112, 112)
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            raise ValueError("Near-zero embedding norm — bad crop?")
        return (emb / norm).astype(np.float32)


def extract_high_quality_frames(
    video_path: str, num_frames: int
) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    n_candidates = num_frames * CANDIDATE_MULTIPLIER
    log.info(f"  {Path(video_path).name} — {total} frames @ {fps:.1f} fps")
    log.info(f"  Scanning {n_candidates} candidates → top {num_frames} by sharpness")

    positions = [
        int(round(i * (total - 1) / (n_candidates - 1)))
        for i in range(n_candidates)
    ]
    candidates = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        score = sharpness_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        candidates.append((score, pos, frame))
    cap.release()

    if not candidates:
        raise RuntimeError("No frames could be read.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:num_frames]
    top.sort(key=lambda x: x[1])
    log.info(
        f"  Sharpness (selected): {top[0][0]:.1f}..{top[-1][0]:.1f}"
        f"  (pool max={candidates[0][0]:.1f})"
    )
    return [(pos, frame) for _, pos, frame in top]


def print_embedding(embedding: np.ndarray, video_name: str):
    print(f"\n  FINAL EMBEDDING — {video_name}  (512 dimensions)")
    print("  " + "─" * 60)
    for i in range(512):
        v = embedding[i]
        print(f"  Dim {i+1:3d}: {'+' if v >= 0 else ''}{v:.8f}")
    print("  " + "─" * 60)
    print(f"  Embedding norm: {np.linalg.norm(embedding):.8f}")
    print("  " + "─" * 60)


def process_video(
    video_path  : str,
    video_index : int,
    video_label : str,
    model       : AdaFaceModel,
    detector    : FaceDetector,
) -> Optional[Tuple[str, np.ndarray]]:

    name      = Path(video_path).name
    video_key = f"video_{video_index}"
    sep       = "─" * 60
    print(f"\n{sep}\n  VIDEO {video_index}: {name}  [{video_label}]\n{sep}")

    frames = extract_high_quality_frames(video_path, FRAMES_TO_USE)
    if not frames:
        log.error("No frames extracted.")
        return None

    crops = []
    for pos, frame in frames:
        crop = detector.detect(frame)
        if crop is not None:
            crops.append((pos, crop))
            log.info(f"  Frame {pos:>5}: face {crop.shape[1]}×{crop.shape[0]}px")
        else:
            log.warning(f"  Frame {pos:>5}: no face — skipped")

    if not crops:
        log.error(f"  No faces found in {name}")
        return None
    log.info(f"  Valid face crops: {len(crops)}/{len(frames)}")

    emb_list  = []
    best_area = 0
    best_masked = None
    for pos, crop in crops:
        resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        masked  = apply_mask(resized)
        emb     = model.get_embedding(masked)
        emb_list.append(emb)
        log.info(f"  Frame {pos:>5}: embedded  norm={np.linalg.norm(emb):.6f}")
        area = crop.shape[0] * crop.shape[1]
        if area > best_area:
            best_area   = area
            best_masked = masked

    out_dir   = Path(OUTPUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"video_{video_index}_masked.jpg"
    cv2.imwrite(str(save_path), best_masked)
    log.info(f"  Masked photo saved → {save_path}")

    stack = np.stack(emb_list, axis=0)
    avg   = np.mean(stack, axis=0).astype(np.float32)
    norm  = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Averaged embedding norm near zero.")
    final = (avg / norm).astype(np.float32)

    visible_rows = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)
    print(f"\n  Frames extracted (high-quality) : {len(frames)}")
    print(f"  Frames with detected face       : {len(crops)}")
    print(f"  Mask cut line                   : row {visible_rows} of 112")
    print(f"    Visible rows  0–{visible_rows-1:<2}           : forehead, eyes, nose, nostrils")
    print(f"    Black   rows  {visible_rows}–111           : mouth, chin")
    print(f"  Saved photo                     : {save_path}")
    print_embedding(final, video_key)
    return video_key, final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two L2-normalised vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if abs(na - 1.0) > 1e-4:
        a = a / na
    if abs(nb - 1.0) > 1e-4:
        b = b / nb
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def compute_and_print_similarity_matrix(
    keys       : List[str],
    labels     : List[str],
    emb_dict   : Dict[str, np.ndarray],
    enroll_key : str,
) -> Dict[str, float]:
    """
    Compute pairwise cosine similarities for all present videos and print:
      1. A full N×N similarity matrix.
      2. A focused table: each video vs the enrolled identity (V1).

    Returns {"{key_a}_vs_{key_b}": similarity} for all pairs (i < j).
    """
    n    = len(keys)
    sims : Dict[str, float] = {}

    # Build full N×N matrix
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(emb_dict[keys[i]], emb_dict[keys[j]])
            matrix[i, j] = matrix[j, i] = sim
            sims[f"{keys[i]}_vs_{keys[j]}"] = sim

    # Short column headers derived from labels: extract tag inside (…)
    short = []
    for lbl in labels:
        if "(" in lbl and ")" in lbl:
            short.append(lbl[lbl.index("(") + 1 : lbl.index(")")])
        else:
            short.append(lbl.strip()[:4])

    col_w = 9    # width per value cell
    lbl_w = 26   # row-label width

    # ── Full N×N matrix ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  COSINE SIMILARITY MATRIX  (all 5 videos)")
    print("═" * 70)
    print()

    header = f"  {'':>{lbl_w}s}  " + "  ".join(f"{s:>{col_w}s}" for s in short)
    print(header)
    divider_len = lbl_w + 2 + (col_w + 2) * n
    print("  " + "─" * divider_len)

    for i in range(n):
        row_lbl  = labels[i][:lbl_w]
        row_vals = []
        for j in range(n):
            if i == j:
                cell = f"{'  1.0000':>{col_w}s}"
            else:
                v      = matrix[i, j]
                marker = "✓" if v >= MIN_SIMILARITY_THRESHOLD else "✗"
                cell   = f"{v:>{col_w - 2}.4f} {marker}"
            row_vals.append(cell)
        print(f"  {row_lbl:>{lbl_w}s}  " + "  ".join(row_vals))

    print()
    print(f"  ✓ = similarity ≥ {MIN_SIMILARITY_THRESHOLD}  "
          f"(same-person pair — PASS expected)")
    print(f"  ✗ = similarity  < {MIN_SIMILARITY_THRESHOLD}  "
          f"(different-person pair — V5 rows/cols expected to show ✗)")

    # ── Per-video vs enrolled identity table ──────────────────────────────────
    enroll_idx = keys.index(enroll_key)
    print()
    print("═" * 70)
    print(f"  COSINE SIMILARITY — each video vs enrolled identity"
          f"  [{labels[enroll_idx].strip()}]")
    print("═" * 70)
    print()
    print(f"  {'Video':<26}  {'Cosine Sim':>10}  {'Δ threshold':>11}  {'Status':>8}")
    print(f"  {'─'*26}  {'─'*10}  {'─'*11}  {'─'*8}")

    non_enroll_sims = []
    for i, (key, lbl) in enumerate(zip(keys, labels)):
        if key == enroll_key:
            continue
        # Retrieve from sims dict regardless of pair ordering
        pair_key = (f"{enroll_key}_vs_{key}"
                    if f"{enroll_key}_vs_{key}" in sims
                    else f"{key}_vs_{enroll_key}")
        sim    = sims.get(pair_key, float(matrix[enroll_idx, i]))
        non_enroll_sims.append(sim)
        delta  = sim - MIN_SIMILARITY_THRESHOLD
        status = "PASS ✓" if sim >= MIN_SIMILARITY_THRESHOLD else "FAIL ✗"
        print(f"  {lbl:<26}  {sim:>10.8f}  {delta:>+11.8f}  {status:>8}")

    print()
    print(f"  Average similarity  (V2–V5 vs V1) : {np.mean(non_enroll_sims):.8f}")
    print(f"  Similarity range    (V2–V5 vs V1) : "
          f"{min(non_enroll_sims):.8f}  –  {max(non_enroll_sims):.8f}")
    print("═" * 70)

    return sims


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GF(2) / GF(2⁸) ARITHMETIC
# ══════════════════════════════════════════════════════════════════════════════

def _gf2_divmod(dividend: list, divisor: list) -> list:
    a = list(dividend); b = list(divisor); db = len(b) - 1
    while len(a) - 1 >= db:
        if a[0] == 1:
            for i in range(len(b)):
                a[i] ^= b[i]
        a.pop(0)
    while len(a) > 1 and a[0] == 0:
        a.pop(0)
    return a


def _poly_pad(poly: list, length: int) -> list:
    p = list(poly)
    while len(p) < length:
        p.insert(0, 0)
    return p[-length:]


def _poly_mul_gf2(a: list, b: list) -> list:
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] ^= (ai & bj)
    while len(result) > 1 and result[0] == 0:
        result.pop(0)
    return result


def _gf256_mul(a: int, b: int, prim: int = 0x11D) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= prim
        b >>= 1
    return result


def _gf256_pow(base: int, exp: int) -> int:
    r = 1
    for _ in range(exp):
        r = _gf256_mul(r, base)
    return r


def _conjugacy_class(exp: int) -> list:
    seen = []; e = exp % 255
    while e not in seen:
        seen.append(e)
        e = (e * 2) % 255
    return seen


def _minimal_poly(root_exp: int) -> list:
    alpha = 2; conj = _conjugacy_class(root_exp); poly = [1]
    for e in conj:
        rv = _gf256_pow(alpha, e)
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i]   ^= c
            new_poly[i+1] ^= _gf256_mul(c, rv)
        poly = new_poly
    return [int(c & 1) for c in poly]


def build_bch_generator(t: int) -> Tuple[list, int, int]:
    """
    Build BCH(255, t) generator polynomial over GF(2).

    Returns (g, K, PAR) where:
        g   = generator polynomial coefficients (0/1 list, MSB first)
        K   = message bits per codeword  = 255 − PAR
        PAR = parity bits               = deg(g)
    """
    g = [1]; used = set()
    for i in range(1, 2 * t, 2):
        cls = frozenset(_conjugacy_class(i))
        if cls in used:
            continue
        used.add(cls)
        g = _poly_mul_gf2(g, _minimal_poly(i))
    PAR = len(g) - 1
    K   = BCH_N - PAR
    return g, K, PAR


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BCH ENCODE / DECODE
# ══════════════════════════════════════════════════════════════════════════════

def bch_encode(msg_bits: list, g: list, K: int, PAR: int) -> list:
    """
    Systematic BCH encode.
    Input : K-bit message.
    Output: 255-bit codeword = [K msg bits | PAR parity bits].
    """
    assert len(msg_bits) == K, f"Expected {K} bits, got {len(msg_bits)}"
    padded    = list(msg_bits) + [0] * PAR
    remainder = _gf2_divmod(padded, g)
    parity    = _poly_pad(remainder, PAR)
    return list(msg_bits) + parity


def bch_decode(
    received_bits: list,
    g            : list,
    K            : int,
    PAR          : int,
    t            : int,
) -> Tuple[list, int]:
    """
    BCH decode via Berlekamp-Massey + Chien search over GF(2⁸).

    Returns (corrected_K_bits, nerr).
    nerr ≥ 0  → success (nerr errors corrected).
    nerr = −1 → failure (errors exceeded t, or Chien root count mismatch).
    """
    assert len(received_bits) == BCH_N

    # GF(2⁸) tables
    GF_EXP = [0] * 512
    GF_LOG = [0] * 256
    x = 1
    for i in range(255):
        GF_EXP[i] = GF_EXP[i + 255] = x
        GF_LOG[x] = i
        x = _gf256_mul(x, 2)

    def gmul(a: int, b: int) -> int:
        return 0 if (a == 0 or b == 0) else GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255]

    def ginv(a: int) -> int:
        return GF_EXP[255 - GF_LOG[a]]

    # Step 1 — 2t syndromes (Horner, MSB-first)
    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = GF_EXP[i % 255]; s = 0
        for bit in received_bits:
            s = gmul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

    # Step 2 — Berlekamp-Massey
    C = [1] + [0] * (2 * t)
    B = [1] + [0] * (2 * t)
    L = 0; m = 1; b = 1

    for n in range(2 * t):
        d = syndromes[n]
        for j in range(1, L + 1):
            if C[j] and syndromes[n - j]:
                d ^= gmul(C[j], syndromes[n - j])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = list(C); coef = gmul(d, ginv(b))
            for j in range(m, 2 * t + 1):
                if j - m < len(B) and B[j - m]:
                    C[j] ^= gmul(coef, B[j - m])
            L = n + 1 - L; B = T; b = d; m = 1
        else:
            coef = gmul(d, ginv(b))
            for j in range(m, 2 * t + 1):
                if j - m < len(B) and B[j - m]:
                    C[j] ^= gmul(coef, B[j - m])
            m += 1

    Lambda = C[:L + 1]
    if L > t or L == 0:
        return list(received_bits[:K]), -1

    # Step 3 — Chien search
    error_positions = []
    for j in range(1, BCH_N + 1):
        val = Lambda[0]
        for k in range(1, len(Lambda)):
            if Lambda[k]:
                val ^= gmul(Lambda[k], GF_EXP[(j * k) % 255])
        if val == 0:
            p = j - 1
            if 0 <= p < BCH_N:
                error_positions.append(p)

    if len(error_positions) != L:
        return list(received_bits[:K]), -1

    # Step 4 — Flip error bits
    corrected = list(received_bits)
    for p in error_positions:
        corrected[p] ^= 1

    return corrected[:K], len(error_positions)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — QUANTISATION
# ══════════════════════════════════════════════════════════════════════════════

def embedding_to_payload(
    emb       : np.ndarray,
    shared_min: Optional[float] = None,
    shared_max: Optional[float] = None,
) -> Tuple[list, np.ndarray, float, float]:
    """
    Quantise 512-dim embedding to QUANT_BITS per dimension.

    With QUANT_BITS=4: 16 levels (0–15) per dimension → 2048-bit payload.

    shared_min/shared_max: must be supplied for all non-enrollment videos
    so that equal float values map to identical bins as V1's enrollment.
    This also applies to V5 (impostor) — same scale, no special treatment.
    """
    levels  = 2 ** QUANT_BITS          # 16 levels for QUANT_BITS=4
    max_val = levels - 1               # 15
    v_min   = shared_min if shared_min is not None else float(emb.min())
    v_max   = shared_max if shared_max is not None else float(emb.max())

    q_vec = np.clip(
        np.round((emb - v_min) / (v_max - v_min) * max_val), 0, max_val
    ).astype(np.int32)

    bits = []
    for q in q_vec:
        for shift in range(QUANT_BITS - 1, -1, -1):
            bits.append(int((int(q) >> shift) & 1))

    return bits, q_vec, v_min, v_max


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bits_to_bytes(bits: list) -> bytes:
    b = list(bits)
    while len(b) % 8 != 0:
        b.insert(0, 0)
    out = bytearray()
    for i in range(0, len(b), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | b[i + j]
        out.append(byte)
    return bytes(out)


def bits_to_hex(bits: list) -> str:
    b = list(bits)
    while len(b) % 4 != 0:
        b.insert(0, 0)
    return "".join(
        format(b[i]*8 + b[i+1]*4 + b[i+2]*2 + b[i+3], "x")
        for i in range(0, len(b), 4)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — FUZZY-COMMITMENT  ENROLL + VERIFY
# ══════════════════════════════════════════════════════════════════════════════

def bch_enroll(
    v1_payload_bits: list,
    g              : list,
    K              : int,
    PAR            : int,
    num_chunks     : int,
) -> Tuple[List[list], str]:
    """
    Enroll V1.

    For each chunk i:
        r[i]       = V1_chunk[i]                (K-bit secret)
        c_r[i]     = BCH_encode(r[i])           (255-bit codeword)
        v1_pad255  = r[i] ++ zeros(PAR)         (extend r to 255 bits)
        helper[i]  = c_r[i] XOR v1_pad255       (stored)

    hash_key = SHA-256(all r bits concatenated)

    Returns
    -------
    helper_data : list of num_chunks 255-bit lists
    hash_key    : 64-char hex SHA-256
    """
    pad      = (num_chunks * K) - len(v1_payload_bits)
    v1_pad   = list(v1_payload_bits) + [0] * pad

    helper_data = []
    all_r_bits  = []

    for i in range(num_chunks):
        r         = v1_pad[i * K : (i + 1) * K]
        c_r       = bch_encode(r, g, K, PAR)
        v1_pad255 = r + [0] * PAR
        helper    = [a ^ b for a, b in zip(c_r, v1_pad255)]
        helper_data.append(helper)
        all_r_bits.extend(r)

    hash_key = hashlib.sha256(bits_to_bytes(all_r_bits)).hexdigest()

    # Self-check: syndromes of each freshly encoded codeword must be zero
    ok = all(
        all(s == 0 for s in _gf2_divmod(bch_encode(v1_pad[i*K:(i+1)*K], g, K, PAR), g))
        for i in range(num_chunks)
    )
    log.info(f"Enrollment — all codeword syndromes zero: {ok}  ← must be True")

    return helper_data, hash_key


def bch_verify(
    vx_payload_bits : list,
    v1_payload_bits : list,
    helper_data     : List[list],
    hash_key_enroll : str,
    g               : list,
    K               : int,
    PAR             : int,
    t               : int,
    num_chunks      : int,
    video_label     : str,
    is_impostor     : bool = False,
) -> dict:
    """
    Verify Vx against enrolled V1.

    Per-chunk:
        noisy[i] = helper[i] XOR (Vx_chunk[i] ++ zeros(PAR))
                 = c_r[i] with error bits at positions where r[i] ≠ Vx_chunk[i]
        BCH_decode(noisy[i]) → r[i]  if  weight(error) ≤ t

    is_impostor is a metadata flag for output labelling only.
    The verification logic is identical for genuine and impostor videos —
    the BCH decoder and hash comparison are the security enforcement.

    Per-chunk tracking:
        t_used[i] = errors corrected by BCH (-1 if chunk failed)
        k_used[i] = Hamming distance between V1 and Vx in chunk i
    """
    pad_vx = (num_chunks * K) - len(vx_payload_bits)
    pad_v1 = (num_chunks * K) - len(v1_payload_bits)
    vx_ext = list(vx_payload_bits) + [0] * pad_vx
    v1_ext = list(v1_payload_bits) + [0] * pad_v1

    identity_tag = "IMPOSTOR" if is_impostor else "GENUINE"

    # ── Hamming diagnostics ──────────────────────────────────────────────────
    total_ham     = sum(a != b for a, b in zip(vx_payload_bits, v1_payload_bits))
    per_chunk_ham = [
        sum(v1_ext[i*K+j] != vx_ext[i*K+j] for j in range(K))
        for i in range(num_chunks)
    ]
    max_chunk_ham   = max(per_chunk_ham)
    chunks_over_t   = sum(1 for e in per_chunk_ham if e > t)
    chunks_within_t = num_chunks - chunks_over_t

    print(f"  Identity tag            : {identity_tag}")
    print(f"  Total bit differences   : {total_ham} / {len(vx_payload_bits)}"
          f"  ({total_ham / len(vx_payload_bits) * 100:.2f}%)")
    print(f"  Max errors in one chunk : {max_chunk_ham}  (BCH limit = {t})")
    print(f"  Chunks within  t={t}   : {chunks_within_t} / {num_chunks}")
    print(f"  Chunks exceeding t={t} : {chunks_over_t}  / {num_chunks}")
    print()
    print("  Per-chunk Hamming distances:")
    for i in range(0, num_chunks, 8):
        row  = per_chunk_ham[i : i + 8]
        line = "  ".join(f"c{i+j:02d}:{row[j]:2d}" for j in range(len(row)))
        print(f"    {line}")

    # ── Fuzzy-commitment correction ──────────────────────────────────────────
    recovered_r = []
    total_corr  = 0
    failed      = 0
    t_used      = []   # nerr per chunk (-1 = failed)
    k_used      = []   # Hamming distance per chunk

    for i in range(num_chunks):
        vx_chunk  = vx_ext[i * K : (i + 1) * K]
        vx_pad255 = vx_chunk + [0] * PAR
        noisy_cw  = [a ^ b for a, b in zip(helper_data[i], vx_pad255)]

        k_used.append(per_chunk_ham[i])

        r_hat, nerr = bch_decode(noisy_cw, g, K, PAR, t)
        t_used.append(nerr)

        if nerr >= 0:
            total_corr += nerr
            recovered_r.extend(r_hat)
        else:
            failed += 1
            recovered_r.extend(vx_chunk)   # fallback → hash mismatch

    # ── Re-hash and compare ───────────────────────────────────────────────────
    hash_verify  = hashlib.sha256(bits_to_bytes(recovered_r)).hexdigest()
    hash_matches = hash_verify == hash_key_enroll

    remaining = sum(
        a != b for a, b in zip(recovered_r, v1_ext[:len(vx_payload_bits)])
    )

    print()
    print(f"  BCH errors corrected    : {total_corr}")
    print(f"  Failed chunks           : {failed}  (0 = full recovery)")
    print(f"  Remaining bit errors    : {remaining}")
    print()
    print(f"  Hash  V1 enroll         : {hash_key_enroll}")
    print(f"  Hash  {video_label} result : {hash_verify}")

    if hash_matches:
        verdict = "PASS  ✓  SAME PERSON — hashes match"
    else:
        if is_impostor:
            verdict = "FAIL  ✗  IMPOSTOR CORRECTLY REJECTED — hashes do NOT match"
        else:
            verdict = "FAIL  ✗  REJECTED — hashes do NOT match"
    print(f"  Result                  : {verdict}")

    # ── Per-chunk t / K usage report ─────────────────────────────────────────
    successful_t = [v for v in t_used if v >= 0]
    failed_idx   = [i for i, v in enumerate(t_used) if v < 0]

    print()
    print("  ── PER-CHUNK t / K USAGE ──────────────────────────────────")
    print(f"  BCH designed t (limit)  : {t}  per chunk")
    print(f"  BCH K (message bits)    : {K}  per chunk")
    print()
    print("  chunk | k_used (hamming) | t_used (corrected)")
    print("  " + "─" * 44)
    for i in range(0, num_chunks, 8):
        k_row = k_used[i : i + 8]
        t_row = t_used[i : i + 8]
        k_line = "  ".join(f"c{i+j:02d}:{k_row[j]:2d}" for j in range(len(k_row)))
        t_line = "  ".join(
            f"c{i+j:02d}:{'F ' if t_row[j] < 0 else str(t_row[j]):>2}"
            for j in range(len(t_row))
        )
        print(f"    k: {k_line}")
        print(f"    t: {t_line}")
        print()

    if successful_t:
        print(f"  t_used (successful chunks only):")
        print(f"    minimum t corrected : {min(successful_t):2d}  "
              f"(chunk c{t_used.index(min(successful_t)):02d})")
        print(f"    maximum t corrected : {max(successful_t):2d}  "
              f"(chunk c{t_used.index(max(successful_t)):02d})")
        print(f"    mean    t corrected : {sum(successful_t)/len(successful_t):.2f}")
    else:
        print("  t_used: no chunks decoded successfully.")

    print()
    print(f"  k_used (Hamming distance per chunk, all {num_chunks} chunks):")
    print(f"    minimum k (easiest chunk) : {min(k_used):2d}  "
          f"(chunk c{k_used.index(min(k_used)):02d})")
    print(f"    maximum k (hardest chunk) : {max(k_used):2d}  "
          f"(chunk c{k_used.index(max(k_used)):02d})")
    print(f"    mean    k                 : {sum(k_used)/len(k_used):.2f}")

    if failed_idx:
        print()
        print(f"  FAILED chunks (t_used = -1, hamming > {t}):")
        for ci in failed_idx:
            print(f"    chunk c{ci:02d}: k_used = {k_used[ci]}  >  t = {t}  ← uncorrectable")
    else:
        print(f"\n  All {num_chunks} chunks corrected successfully.")

    print("  " + "─" * 55)
    print("─" * 60)

    return {
        "label"        : video_label,
        "is_impostor"  : is_impostor,
        "hamming_total": total_ham,
        "max_chunk"    : max_chunk_ham,
        "chunks_over_t": chunks_over_t,
        "failed_chunks": failed,
        "corrected"    : total_corr,
        "remaining"    : remaining,
        "hash_matches" : hash_matches,
        "t_used"       : t_used,
        "k_used"       : k_used,
        "t_min"        : min(successful_t) if successful_t else None,
        "t_max"        : max(successful_t) if successful_t else None,
        "k_min"        : min(k_used),
        "k_max"        : max(k_used),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    sep = "═" * 70

    # Build BCH generator and verify K / PAR match the specified parameters
    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial …")
    g, derived_K, derived_PAR = build_bch_generator(BCH_T_DESIGNED)

    if derived_K != BCH_K or derived_PAR != BCH_PAR:
        log.warning(
            f"Generator polynomial gives K={derived_K}, PAR={derived_PAR} "
            f"but config specifies K={BCH_K}, PAR={BCH_PAR}. "
            f"Using derived values."
        )
    K   = derived_K
    PAR = derived_PAR

    PAYLOAD_BITS = 512 * QUANT_BITS                    # 2048
    num_chunks   = math.ceil(PAYLOAD_BITS / K)         # 29
    t_total      = num_chunks * BCH_T_DESIGNED         # 725
    pad_needed   = num_chunks * K - PAYLOAD_BITS       # 11
    VIS_ROWS     = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)

    # Keys and labels derived entirely from VIDEO_PATHS / VIDEO_LABELS config
    video_keys   = [f"video_{i+1}" for i in range(len(VIDEO_PATHS))]
    enroll_key   = video_keys[ENROLL_INDEX]
    impostor_key = video_keys[IMPOSTOR_INDEX]

    print(sep)
    print("  ADAFACE EMBEDDING PIPELINE  +  BCH FUZZY-COMMITMENT")
    print("  Phase 11  —  QUANT_BITS=4  +  V5 IMPOSTOR TEST")
    print(sep)
    print(f"  Videos             : {len(VIDEO_PATHS)}")
    for i, (vp, lbl) in enumerate(zip(VIDEO_PATHS, VIDEO_LABELS), 1):
        tag = " ← ENROLL" if i - 1 == ENROLL_INDEX else (
              " ← IMPOSTOR" if i - 1 == IMPOSTOR_INDEX else "")
        print(f"    {i}. {Path(vp).name:<46} [{lbl}]{tag}")
    print(f"  Frame strategy     : top {FRAMES_TO_USE} sharpest of "
          f"{FRAMES_TO_USE * CANDIDATE_MULTIPLIER} candidates")
    print(f"  Mask               : rows {VIS_ROWS}–111 black  |  rows 0–{VIS_ROWS-1} visible")
    print()
    print("  BCH PARAMETERS  —  Phase 11")
    print(f"    QUANT_BITS      : {QUANT_BITS}  →  payload = 512×{QUANT_BITS} = {PAYLOAD_BITS} bits")
    print(f"    Quant levels    : 2^{QUANT_BITS} = {2**QUANT_BITS} levels per dimension (0–{2**QUANT_BITS - 1})")
    print(f"    BCH code        : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED})")
    print(f"    Parity bits     : {PAR}  (deg of generator polynomial, verified)")
    print(f"    Chunks          : {num_chunks}  (last chunk zero-padded {pad_needed} bits)")
    print(f"    t per chunk     : {BCH_T_DESIGNED}")
    print(f"    t TOTAL         : {t_total}  (= {num_chunks} × {BCH_T_DESIGNED})")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    # ── Process all 5 videos ─────────────────────────────────────────────────
    embeddings: Dict[str, np.ndarray] = {}
    for idx, (vp, lbl) in enumerate(zip(VIDEO_PATHS, VIDEO_LABELS), start=1):
        result = process_video(vp, idx, lbl, model, detector)
        if result:
            key, emb = result
            embeddings[key] = emb

    print(f"\n{sep}")
    print("  PROCESSING COMPLETE — FINAL EMBEDDING NORMS")
    print(sep)
    for key, lbl in zip(video_keys, VIDEO_LABELS):
        if key in embeddings:
            norm = np.linalg.norm(embeddings[key])
            tag  = " ← enrolled" if key == enroll_key else (
                   " ← impostor" if key == impostor_key else "")
            print(f"  {lbl:<26}  norm = {norm:.8f}{tag}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep)

    # ── Cosine similarity matrix (all 5 videos) ───────────────────────────────
    present_keys   = [k for k in video_keys if k in embeddings]
    present_labels = [VIDEO_LABELS[video_keys.index(k)] for k in present_keys]

    similarities: Dict[str, float] = {}
    if len(present_keys) >= 2:
        similarities = compute_and_print_similarity_matrix(
            keys       = present_keys,
            labels     = present_labels,
            emb_dict   = embeddings,
            enroll_key = enroll_key,
        )

    # ── BCH Fuzzy-Commitment ──────────────────────────────────────────────────
    if enroll_key not in embeddings:
        print(f"\n{sep}\n  ERROR: enrollment video unavailable — cannot enroll.\n{sep}")
        return embeddings, similarities

    print(f"\n{'='*70}")
    print("  PHASE 11 — BCH FUZZY-COMMITMENT")
    print("  ENROLL V1  /  VERIFY V2, V3, V4 (genuine)  /  V5 (impostor)")
    print(f"{'='*70}")
    print(f"  BCH code   : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED}) × {num_chunks} chunks")
    print(f"  t total    : {t_total}")
    print(f"  Payload    : {PAYLOAD_BITS} bits  ({QUANT_BITS}-bit quant × 512 dims)")
    print(f"  Scale      : V1's [v_min, v_max] shared by all videos including V5")

    # Quantise V1 — establishes shared scale for ALL subsequent videos
    v1_bits, _, v1_min, v1_max = embedding_to_payload(embeddings[enroll_key])
    log.info(f"V1 quantised — scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
             f"payload length = {len(v1_bits)} bits")

    # ── ENROLLMENT ────────────────────────────────────────────────────────────
    enroll_label = VIDEO_LABELS[ENROLL_INDEX]
    print(f"\n{'─'*60}")
    print(f"  ENROLLMENT — {enroll_label}")
    print(f"{'─'*60}")

    helper_data, hash_key_H1 = bch_enroll(
        v1_payload_bits=v1_bits,
        g=g, K=K, PAR=PAR,
        num_chunks=num_chunks,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Payload bits         : {len(v1_bits)}")
    print(f"  Chunks enrolled      : {num_chunks}")
    print(f"  Helper size          : {num_chunks} × {BCH_N} = {num_chunks * BCH_N} bits")
    print(f"  Hash key (SHA-256)   : {hash_key_H1}")
    print(f"  Helper (first 64 hex): {helper_hex[:64]}…")
    print(f"  V1 scale             : [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"{'─'*60}")

    # ── VERIFICATION — all non-enrolled videos (V2, V3, V4, V5) ─────────────
    # Derived entirely from config — no video-name strings hardcoded here.
    verify_entries = [
        (video_keys[i], VIDEO_LABELS[i], i == IMPOSTOR_INDEX)
        for i in range(len(VIDEO_PATHS))
        if i != ENROLL_INDEX
    ]

    summary = []
    for vid_key, vid_label, is_imp in verify_entries:
        if vid_key not in embeddings:
            log.error(f"{vid_key} ({vid_label}) not available — skipping.")
            continue

        print(f"\n{'─'*60}")
        imp_tag = "  ← IMPOSTOR TEST" if is_imp else ""
        print(f"  VERIFICATION — {vid_label}{imp_tag}")
        print(f"{'─'*60}")

        vx_bits, _, _, _ = embedding_to_payload(
            embeddings[vid_key], shared_min=v1_min, shared_max=v1_max
        )
        log.info(f"{vid_key} quantised using V1 scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
                 f"payload length = {len(vx_bits)} bits")

        result = bch_verify(
            vx_payload_bits = vx_bits,
            v1_payload_bits = v1_bits,
            helper_data     = helper_data,
            hash_key_enroll = hash_key_H1,
            g=g, K=K, PAR=PAR, t=BCH_T_DESIGNED,
            num_chunks      = num_chunks,
            video_label     = vid_key,
            is_impostor     = is_imp,
        )
        summary.append(result)

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 11 FINAL SUMMARY  —  V1 enrolled  vs  V2 / V3 / V4 / V5")
    print(f"{'='*70}")
    print(f"  Enrollment   : {VIDEO_LABELS[ENROLL_INDEX]}")
    print(f"  Hash key H1  : {hash_key_H1}")
    print(f"  BCH params   : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED})"
          f" × {num_chunks} chunks  (t_total={t_total})")
    print(f"  Shared scale : V1 range [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"  QUANT_BITS   : {QUANT_BITS}  ({2**QUANT_BITS} levels, payload={PAYLOAD_BITS} bits)")
    print()
    print(f"  {'Video':<26}  {'Type':<9}  {'Ham-Tot':>7}  {'Max/Chk':>7}  "
          f"{'Chk>t':>5}  {'Failed':>6}  {'Corrctd':>7}  {'Expected':>8}  {'Result'}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*7}  {'─'*7}  "
          f"{'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*15}")

    for r in summary:
        v_type   = "IMPOSTOR" if r["is_impostor"] else "GENUINE"
        expected = "FAIL" if r["is_impostor"] else "PASS"
        actual   = "PASS" if r["hash_matches"] else "FAIL"
        correct  = "✓" if actual == expected else "✗ UNEXPECTED"
        print(
            f"  {r['label']:<26}  "
            f"{v_type:<9}  "
            f"{r['hamming_total']:>7}  "
            f"{r['max_chunk']:>7}  "
            f"{r['chunks_over_t']:>5}  "
            f"{r['failed_chunks']:>6}  "
            f"{r['corrected']:>7}  "
            f"{expected:>8}  "
            f"{actual} {correct}"
        )

    # ── T AND K USAGE ACROSS ALL VERIFICATIONS ────────────────────────────────
    print()
    print(f"  {'═'*68}")
    print("  T AND K USAGE ACROSS ALL VERIFICATIONS  (V2 – V5)")
    print(f"  {'═'*68}")
    print(f"  BCH designed t = {BCH_T_DESIGNED}  (errors correctable per chunk)")
    print(f"  BCH K          = {K}  (message bits per chunk)")
    print()
    print(f"  {'Video':<26}  {'Type':<9}  {'k_min':>5}  {'k_max':>5}  {'k_mean':>6}  "
          f"{'t_min':>5}  {'t_max':>5}  {'t_mean':>6}  {'t_failed':>8}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*6}  "
          f"{'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}")

    all_k_vals : List[int] = []
    all_t_vals : List[int] = []

    for r in summary:
        k_vals = r["k_used"]
        t_vals = [v for v in r["t_used"] if v >= 0]
        t_fail = sum(1 for v in r["t_used"] if v < 0)
        v_type = "IMPOSTOR" if r["is_impostor"] else "GENUINE"

        k_min  = min(k_vals)
        k_max  = max(k_vals)
        k_mean = sum(k_vals) / len(k_vals)
        t_min  = min(t_vals) if t_vals else None
        t_max  = max(t_vals) if t_vals else None
        t_mean = sum(t_vals) / len(t_vals) if t_vals else None

        all_k_vals.extend(k_vals)
        all_t_vals.extend(t_vals)

        t_min_s  = f"{t_min:>5}" if t_min is not None else "  N/A"
        t_max_s  = f"{t_max:>5}" if t_max is not None else "  N/A"
        t_mean_s = f"{t_mean:>6.2f}" if t_mean is not None else "   N/A"

        print(
            f"  {r['label']:<26}  "
            f"{v_type:<9}  "
            f"{k_min:>5}  {k_max:>5}  {k_mean:>6.2f}  "
            f"{t_min_s}  {t_max_s}  {t_mean_s}  "
            f"{t_fail:>8}"
        )

    # Overall row across all four verification videos
    if all_k_vals:
        print(f"  {'─'*26}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*6}  "
              f"{'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}")
        all_t_min  = min(all_t_vals) if all_t_vals else None
        all_t_max  = max(all_t_vals) if all_t_vals else None
        all_t_mean = sum(all_t_vals) / len(all_t_vals) if all_t_vals else None
        all_k_min  = min(all_k_vals)
        all_k_max  = max(all_k_vals)
        all_k_mean = sum(all_k_vals) / len(all_k_vals)
        all_t_fail = sum(1 for r in summary for v in r["t_used"] if v < 0)

        t_min_s  = f"{all_t_min:>5}" if all_t_min is not None else "  N/A"
        t_max_s  = f"{all_t_max:>5}" if all_t_max is not None else "  N/A"
        t_mean_s = f"{all_t_mean:>6.2f}" if all_t_mean is not None else "   N/A"

        print(
            f"  {'OVERALL (V2–V5)':<26}  "
            f"{'ALL':<9}  "
            f"{all_k_min:>5}  {all_k_max:>5}  {all_k_mean:>6.2f}  "
            f"{t_min_s}  {t_max_s}  {t_mean_s}  "
            f"{all_t_fail:>8}"
        )

    print()
    print("  COLUMN GUIDE")
    print(f"  k_min / k_max  : min / max Hamming distance between V1 and Vx per chunk")
    print(f"  k_mean         : average Hamming distance per chunk")
    print(f"  t_min / t_max  : min / max errors corrected by BCH per chunk")
    print(f"                   (successful chunks only — failed = -1 excluded)")
    print(f"  t_mean         : average errors corrected per successful chunk")
    print(f"  t_failed       : chunks where BCH decoder returned -1 (exceeded t={BCH_T_DESIGNED})")
    print(f"  BCH hard limit : t = {BCH_T_DESIGNED} errors per chunk  |  K = {K} message bits")
    print(f"  {'═'*68}")

    # ── OUTCOME ───────────────────────────────────────────────────────────────
    print()
    genuine_results  = [r for r in summary if not r["is_impostor"]]
    impostor_results = [r for r in summary if r["is_impostor"]]

    genuine_all_pass  = all(r["hash_matches"] for r in genuine_results)
    impostor_all_fail = all(not r["hash_matches"] for r in impostor_results)

    g_pass  = sum(1 for r in genuine_results if r["hash_matches"])
    g_total = len(genuine_results)
    i_fail  = sum(1 for r in impostor_results if not r["hash_matches"])
    i_total = len(impostor_results)

    print("  OUTCOME SUMMARY")
    print(f"  {'─'*54}")
    if genuine_results:
        print(f"  Genuine  (V2–V4) : {g_pass}/{g_total} PASSED"
              f"  {'✓ ALL CORRECT' if genuine_all_pass else '✗ UNEXPECTED FAILURES'}")
    if impostor_results:
        print(f"  Impostor (V5)    : {i_fail}/{i_total} correctly REJECTED"
              f"  {'✓ SECURITY HOLDS' if impostor_all_fail else '✗ IMPOSTOR ACCEPTED — BREACH'}")

    print()
    if genuine_all_pass and impostor_all_fail:
        print("  ✓  SYSTEM WORKING CORRECTLY")
        print("     Same-person videos PASS  ·  Impostor video correctly REJECTED")
    elif not genuine_all_pass and impostor_all_fail:
        print("  ⚠  PARTIAL FAILURE: some genuine videos rejected")
        print("     Impostor correctly rejected — consider raising BCH_T_DESIGNED")
    elif genuine_all_pass and not impostor_all_fail:
        print("  ✗  SECURITY BREACH: impostor accepted")
        print("     Consider lowering QUANT_BITS or reviewing BCH parameters")
    else:
        print("  ✗  SYSTEM FAILURE: multiple issues detected")

    print(f"{'='*70}")
    return embeddings, similarities


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY-COMMITMENT  —  Phase 11")
    print("  QUANT_BITS=4  |  t=25  |  K=71  |  29 chunks  |  V5 impostor test")
    print("=" * 70)

    embeddings, pairwise_sims = run()

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Videos processed  : {len(embeddings)} / {len(VIDEO_PATHS)}")
    print(f"  Pairs compared    : {len(pairwise_sims)}")

    if pairwise_sims:
        sv = list(pairwise_sims.values())
        print(f"  Average similarity (all pairs) : {np.mean(sv):.8f}")
        print(f"  Similarity range   (all pairs) : {min(sv):.8f} – {max(sv):.8f}")

    print("=" * 70)
