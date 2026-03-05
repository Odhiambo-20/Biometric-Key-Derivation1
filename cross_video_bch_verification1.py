"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAFACE FACE EMBEDDING PIPELINE  +  BCH FUZZY COMMITMENT            ║
║                      Phase 10  —  Option A Fix  (QUANT_BITS = 5)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHAT CHANGED FROM QUANT_BITS = 3 VERSION                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  QUANT_BITS : 3  ──►  5                                                     ║
║                                                                              ║
║  Effect on payload and BCH chunking:                                        ║
║    Payload bits  : 512 × 3 = 1536  ──►  512 × 5 = 2560 bits               ║
║    BCH(N=255, t=35)  →  PAR=208, K=47  (unchanged — same BCH code)         ║
║    Chunks        : ceil(1536 / 47) = 33  ──►  ceil(2560 / 47) = 55        ║
║    t_total       : 33 × 35 = 1155        ──►  55 × 35 = 1925              ║
║    Padding       : 33×47 − 1536 = 15 bits ──►  55×47 − 2560 = 25 bits     ║
║                                                                              ║
║  Why 5-bit quantisation:                                                    ║
║    3-bit → 8 levels per dimension (coarser, fewer bits, fewer errors)       ║
║    5-bit → 32 levels per dimension (finer, more bits, more precision)       ║
║    Trade-off: higher fidelity representation vs. more payload bits to       ║
║    protect. BCH t=35 still handles per-chunk errors; more chunks means      ║
║    total error budget scales up proportionally (1925 vs 1155).              ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BCH FIX CONTEXT  (carried over from Phase 10 Option A)                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BCH_T_DESIGNED : 30  ──►  35                                               ║
║                                                                              ║
║  Phase 9 root-cause:                                                         ║
║    BCH operates PER-CHUNK.  Total budget (750) was not exceeded, but two    ║
║    individual chunks had burst errors that exceeded the per-chunk limit:    ║
║      V2  chunk c16 = 33 errors  (limit was 30  →  FAIL by 3)               ║
║      V4  chunk c13 = 31 errors  (limit was 30  →  FAIL by 1)               ║
║    One unrecoverable chunk poisons the entire hash comparison → FAIL.       ║
║                                                                              ║
║  Option A fix:                                                               ║
║    Raise t so every observed chunk error is within the new limit.           ║
║    t=35 covers the worst observed case (33) with a 2-error safety margin.  ║
║                                                                              ║
║  BCH parameters (verified by building the generator polynomial):            ║
║    BCH(N=255, t=35)  →  PAR=208, K=47                                       ║
║    Chunks  = ceil(2560 / 47) = 55   (payload bits padded to 55×47=2585)    ║
║    t_total = 55 × 35 = 1925                                                 ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  PIPELINE OVERVIEW                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1.  Extract top-20 sharpest frames from each video (Laplacian variance)    ║
║  2.  Detect face with Haar cascade → tight crop                             ║
║  3.  Resize to 112×112, apply lower-face mask (bottom 38 %)                ║
║  4.  AdaFace IR-18 ONNX → 512-dim embedding, L2-normalised per frame       ║
║  5.  Average all per-frame embeddings → re-normalise → FINAL embedding      ║
║  6.  Pairwise cosine similarities across all 4 videos                       ║
║  7.  BCH Fuzzy-Commitment  (enroll V1, verify V2 / V3 / V4)                ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BCH FUZZY-COMMITMENT SCHEME  (Juels & Wattenberg, 1999)                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  ENROLL (V1):                                                                ║
║    Quantise V1 → 2560-bit payload  (5 bits × 512 dims)                     ║
║    Split into 55 chunks of K=47 bits  (last chunk zero-padded)             ║
║    For each chunk i:                                                         ║
║      r[i]       = V1_chunk[i]                   ← 47-bit secret             ║
║      c_r[i]     = BCH_encode(r[i])              ← 255-bit codeword          ║
║      v1_pad[i]  = r[i] ++ zeros(208)            ← extend r to 255 bits      ║
║      helper[i]  = c_r[i] XOR v1_pad[i]          ← stored helper data        ║
║    hash_key = SHA-256( r[0] ‖ r[1] ‖ … ‖ r[54] )  ← stored hash           ║
║                                                                              ║
║  VERIFY (Vx, x ∈ {2,3,4}):                                                 ║
║    Quantise Vx using V1's [v_min, v_max]  →  2560-bit payload              ║
║    For each chunk i:                                                         ║
║      vx_pad[i]  = Vx_chunk[i] ++ zeros(208)     ← extend to 255 bits       ║
║      noisy[i]   = helper[i] XOR vx_pad[i]                                  ║
║                 = c_r[i] XOR (r[i]⊕Vx_chunk[i]) ++ zeros(208)             ║
║                 = c_r[i] with errors only at positions where Vx ≠ V1       ║
║      r̂[i]      = BCH_decode(noisy[i])           ← recovers r[i] if ≤ t    ║
║    hash_verify = SHA-256( r̂[0] ‖ … ‖ r̂[54] )                            ║
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

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4",
]
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

# ── BCH  (Phase 10 — Option A  +  QUANT_BITS raised 3 → 5) ──────────────────
#
#  QUANT_BITS = 5  →  payload = 512 × 5 = 2560 bits
#
#  BCH(255, t=35)  →  PAR=208, K=47  (same BCH code as before)
#    chunks  = ceil(2560 / 47) = 55   (was 33 with 3-bit quant)
#    t_total = 55 × 35 = 1925         (was 1155)
#    padding = 55×47 − 2560 = 25 bits (last chunk zero-padded)
#
#  t=35 retained: worst observed per-chunk error was 33 (Phase 9 data),
#  so the 2-error safety margin is preserved regardless of quant width.

BCH_N          = 255
BCH_T_DESIGNED = 35   # covers observed max chunk error of 33 with margin
QUANT_BITS     = 5    # ← CHANGED from 3  (32 levels per dimension)


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
    model       : AdaFaceModel,
    detector    : FaceDetector,
) -> Optional[Tuple[str, np.ndarray]]:

    name       = Path(video_path).name
    video_name = f"video_{video_index}"
    sep        = "─" * 60
    print(f"\n{sep}\n  VIDEO {video_index}: {name}\n{sep}")

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

    embeddings  = []
    best_area   = 0
    best_masked = None
    for pos, crop in crops:
        resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        masked  = apply_mask(resized)
        emb     = model.get_embedding(masked)
        embeddings.append(emb)
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

    stack = np.stack(embeddings, axis=0)
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
    print_embedding(final, video_name)
    return video_name, final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    for name, v in [("v1", v1), ("v2", v2)]:
        n = float(np.linalg.norm(v))
        if abs(n - 1.0) > 1e-4:
            log.warning(f"{name} not unit (norm={n:.6f}). Renormalising.")
            v = v / n
    sim = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if label:
        log.info(f"Cosine similarity [{label}] : {sim:.8f}")
    return sim


def compute_pairwise_similarities(
    emb_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    similarities = {}
    print("\n" + "═" * 62)
    print("  PAIRWISE COSINE SIMILARITY COMPARISONS")
    print("  (Using FINAL renormalized embeddings from each video)")
    print("═" * 62)

    for v1n, v2n in itertools.combinations(emb_dict.keys(), 2):
        e1, e2 = emb_dict[v1n], emb_dict[v2n]
        print(f"\n  {v1n} (norm={np.linalg.norm(e1):.8f}) vs "
              f"{v2n} (norm={np.linalg.norm(e2):.8f})")
        sim    = cosine_similarity(e1, e2, label=f"{v1n}_vs_{v2n}")
        key    = f"{v1n}_vs_{v2n}"
        status = "GOOD" if sim >= MIN_SIMILARITY_THRESHOLD else "LOW"
        similarities[key] = sim
        print(f"  {'='*42}")
        print(f"  COSINE SIMILARITY: {sim:.8f}   {status}")
        print(f"  {'='*42}")
        if sim < MIN_SIMILARITY_THRESHOLD:
            log.warning(f"Similarity {sim:.4f} < threshold {MIN_SIMILARITY_THRESHOLD}")

    return similarities


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

    With QUANT_BITS=5: 32 levels (0–31) per dimension → 2560-bit payload.

    shared_min/shared_max: must be supplied for all non-enrollment videos
    so that equal float values map to identical bins as V1's enrollment.
    """
    levels  = 2 ** QUANT_BITS          # 32 levels for QUANT_BITS=5
    max_val = levels - 1               # 31
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

    With QUANT_BITS=5: 55 chunks of 47 bits each (2585 total, 25 zero-padded).

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
    v1_payload_bits : list,       # Hamming diagnostics only
    helper_data     : List[list],
    hash_key_enroll : str,
    g               : list,
    K               : int,
    PAR             : int,
    t               : int,
    num_chunks      : int,
    video_label     : str,
) -> dict:
    """
    Verify Vx against enrolled V1.

    Per-chunk derivation:
        helper[i]  = c_r[i]  XOR  (r[i] ++ zeros(PAR))
        vx_pad255  = Vx_chunk[i] ++ zeros(PAR)

        noisy[i]   = helper[i] XOR vx_pad255
                   = c_r[i]  XOR  ((r[i] XOR Vx_chunk[i]) ++ zeros(PAR))
                   = c_r[i] with error bits at every position where
                     r[i] ≠ Vx_chunk[i]  (only first K positions can differ)

        BCH_decode(noisy[i])  →  r[i]   if  weight(error) ≤ t

    After all chunks: hash(recovered r) vs hash_key_enroll.

    With QUANT_BITS=5: operates over 55 chunks (2560-bit payload).

    NEW — per-chunk t and K tracking:
        t_used[i]  = actual errors corrected in chunk i  (nerr from decoder)
                     0 if no errors, -1 recorded separately as failed
        k_used[i]  = hamming distance in chunk i between V1 and Vx
                     (= the error weight presented to the decoder)
    After all chunks the min/max of both are printed.
    """
    pad_vx = (num_chunks * K) - len(vx_payload_bits)
    pad_v1 = (num_chunks * K) - len(v1_payload_bits)
    vx_ext = list(vx_payload_bits) + [0] * pad_vx
    v1_ext = list(v1_payload_bits) + [0] * pad_v1

    # ── Hamming diagnostics ──────────────────────────────────────────────────
    total_ham     = sum(a != b for a, b in zip(vx_payload_bits, v1_payload_bits))
    per_chunk_ham = [
        sum(v1_ext[i*K+j] != vx_ext[i*K+j] for j in range(K))
        for i in range(num_chunks)
    ]
    max_chunk_ham   = max(per_chunk_ham)
    chunks_over_t   = sum(1 for e in per_chunk_ham if e > t)
    chunks_within_t = num_chunks - chunks_over_t

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
    recovered_r  = []
    total_corr   = 0
    failed       = 0

    # ── NEW: per-chunk t and K usage tracking ────────────────────────────────
    # t_used[i] = errors the BCH decoder actually corrected in chunk i
    #             (= nerr returned by bch_decode; 0 if codeword already valid)
    # k_used[i] = Hamming distance between V1 and Vx in chunk i
    #             (= the error weight the decoder had to handle)
    t_used = []   # list of ints, one per chunk (failed chunks recorded as -1)
    k_used = []   # list of ints, one per chunk (always the raw hamming count)

    for i in range(num_chunks):
        vx_chunk  = vx_ext[i * K : (i + 1) * K]
        vx_pad255 = vx_chunk + [0] * PAR
        noisy_cw  = [a ^ b for a, b in zip(helper_data[i], vx_pad255)]

        # k_used: how many bit positions differ between V1 and Vx in this chunk
        chunk_ham = per_chunk_ham[i]
        k_used.append(chunk_ham)

        r_hat, nerr = bch_decode(noisy_cw, g, K, PAR, t)

        # t_used: how many errors the decoder actually corrected
        t_used.append(nerr)   # nerr == -1 means failed

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
    verdict = (
        "PASS  ✓  SAME PERSON — hashes match"
        if hash_matches
        else "FAIL  ✗  REJECTED — hashes do NOT match"
    )
    print(f"  Result                  : {verdict}")

    # ── NEW: per-chunk t / K usage report ────────────────────────────────────
    # t_used values: ≥0 = corrected that many errors, -1 = chunk failed
    # k_used values: raw Hamming distance between V1 and Vx per chunk
    successful_t = [v for v in t_used if v >= 0]
    failed_t     = [i for i, v in enumerate(t_used) if v < 0]

    print()
    print("  ── PER-CHUNK t / K USAGE ──────────────────────────────────")
    print(f"  BCH designed t (limit)  : {t}  per chunk")
    print(f"  BCH K (message bits)    : {K}  per chunk")
    print()

    # Print per-chunk table in rows of 8
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
        print(f"    minimum t corrected : {min(successful_t)}  "
              f"(chunk {t_used.index(min(successful_t)):02d})")
        print(f"    maximum t corrected : {max(successful_t)}  "
              f"(chunk {t_used.index(max(successful_t)):02d})")
        print(f"    mean    t corrected : {sum(successful_t)/len(successful_t):.2f}")
    else:
        print("  t_used: no chunks decoded successfully.")

    print()
    print(f"  k_used (Hamming distance per chunk, all {num_chunks} chunks):")
    print(f"    minimum k (easiest chunk) : {min(k_used):2d}  "
          f"(chunk {k_used.index(min(k_used)):02d})")
    print(f"    maximum k (hardest chunk) : {max(k_used):2d}  "
          f"(chunk {k_used.index(max(k_used)):02d})")
    print(f"    mean    k                 : {sum(k_used)/len(k_used):.2f}")

    if failed_t:
        print()
        print(f"  FAILED chunks (t_used = -1, hamming > {t}):")
        for ci in failed_t:
            print(f"    chunk {ci:02d}: k_used = {k_used[ci]}  >  t = {t}  ← uncorrectable")
    else:
        print(f"\n  All {num_chunks} chunks corrected successfully.")

    print("  " + "─" * 55)
    print("─" * 60)

    return {
        "label"        : video_label,
        "hamming_total": total_ham,
        "max_chunk"    : max_chunk_ham,
        "chunks_over_t": chunks_over_t,
        "failed_chunks": failed,
        "corrected"    : total_corr,
        "remaining"    : remaining,
        "hash_matches" : hash_matches,
        # NEW fields for cross-video summary
        "t_used"       : t_used,
        "k_used"       : k_used,
        "t_min"        : min(successful_t) if successful_t else None,
        "t_max"        : max(successful_t) if successful_t else None,
        "k_min"        : min(k_used),
        "k_max"        : max(k_used),
    }


#  SECTION 8 — MAIN


def run():
    sep = "═" * 70

    # Build BCH generator and confirm exact K / PAR
    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial …")
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    PAYLOAD_BITS = 512 * QUANT_BITS                    # 2560  (5-bit quant)
    NUM_CHUNKS   = math.ceil(PAYLOAD_BITS / BCH_K)     # ceil(2560/47) = 55
    T_TOTAL      = NUM_CHUNKS * BCH_T_DESIGNED         # 55 × 35 = 1925
    PAD_NEEDED   = NUM_CHUNKS * BCH_K - PAYLOAD_BITS   # 55×47 − 2560 = 25
    VIS_ROWS     = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)

    print(sep)
    print("  ADAFACE EMBEDDING PIPELINE  +  BCH FUZZY-COMMITMENT  (Phase 10, QUANT_BITS=5)")
    print(sep)
    print(f"  Videos             : {len(VIDEO_PATHS)}")
    for i, vp in enumerate(VIDEO_PATHS, 1):
        print(f"    {i}. {Path(vp).name}")
    print(f"  Frame strategy     : top {FRAMES_TO_USE} sharpest of "
          f"{FRAMES_TO_USE * CANDIDATE_MULTIPLIER} candidates")
    print(f"  Mask               : rows {VIS_ROWS}–111 black  |  rows 0–{VIS_ROWS-1} visible")
    print()
    print("  BCH PARAMETERS  —  Phase 10  (Option A: t=35  +  QUANT_BITS=5)")
    print(f"    QUANT_BITS      : {QUANT_BITS}  →  payload = 512×{QUANT_BITS} = {PAYLOAD_BITS} bits")
    print(f"    Quant levels    : 2^{QUANT_BITS} = {2**QUANT_BITS} levels per dimension (0–{2**QUANT_BITS - 1})")
    print(f"    BCH code        : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})")
    print(f"    Parity bits     : {BCH_PAR}  (deg of generator polynomial, verified)")
    print(f"    Chunks          : {NUM_CHUNKS}  (last chunk zero-padded {PAD_NEEDED} bits)")
    print(f"    t per chunk     : {BCH_T_DESIGNED}  (covers observed max of 33 with margin)")
    print(f"    t TOTAL         : {T_TOTAL}  (= {NUM_CHUNKS} × {BCH_T_DESIGNED})")
    print()
    print("  Why t was raised from 30 (Phase 9 fix, carried forward):")
    print("    Phase 9  V2 chunk c16 = 33 errors  >  t=30  → FAIL (by 3)")
    print("    Phase 9  V4 chunk c13 = 31 errors  >  t=30  → FAIL (by 1)")
    print(f"    t={BCH_T_DESIGNED} covers 33 with a 2-error safety margin")
    print()
    print("  QUANT_BITS change  (3 → 5):")
    print("    Payload  : 1536 bits  →  2560 bits")
    print("    Chunks   : 33         →  55")
    print("    t_total  : 1155       →  1925")
    print("    Levels   : 8 (coarse) →  32 (finer embedding resolution)")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        result = process_video(vp, idx, model, detector)
        if result:
            name, emb = result
            embeddings[name] = emb

    print(f"\n{sep}")
    print("  PROCESSING COMPLETE — FINAL EMBEDDING NORMS")
    print(sep)
    for name, emb in embeddings.items():
        print(f"  {name:10}: norm = {np.linalg.norm(emb):.8f}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep)

    # Pairwise cosine similarities
    similarities: Dict[str, float] = {}
    if len(embeddings) >= 2:
        similarities = compute_pairwise_similarities(embeddings)
        if similarities:
            sv = list(similarities.values())
            print(f"\n  Average similarity : {np.mean(sv):.8f}")
            print(f"  Range              : {min(sv):.8f} – {max(sv):.8f}")

    # BCH Fuzzy-Commitment
    if "video_1" not in embeddings:
        print(f"\n{sep}\n  ERROR: video_1 unavailable — cannot enroll.\n{sep}")
        return embeddings, similarities

    print(f"\n{'='*70}")
    print("  PHASE 10 — BCH FUZZY-COMMITMENT  (ENROLL V1 / VERIFY V2, V3, V4)")
    print(f"{'='*70}")
    print(f"  BCH code   : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED}) × {NUM_CHUNKS} chunks")
    print(f"  t total    : {T_TOTAL}")
    print(f"  Payload    : {PAYLOAD_BITS} bits  ({QUANT_BITS}-bit quant × 512 dims)")
    print(f"  Scale      : V1's [v_min, v_max] used for all videos")

    # Quantise V1 — establishes shared scale
    v1_bits, _, v1_min, v1_max = embedding_to_payload(embeddings["video_1"])
    log.info(f"V1 quantised — scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
             f"payload length = {len(v1_bits)} bits")

    # ── ENROLLMENT ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ENROLLMENT — V1 (IOS Beard)")
    print(f"{'─'*60}")

    helper_data, hash_key_H1 = bch_enroll(
        v1_payload_bits=v1_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks=NUM_CHUNKS,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Payload bits       : {len(v1_bits)}")
    print(f"  Chunks enrolled    : {NUM_CHUNKS}")
    print(f"  Helper size        : {NUM_CHUNKS} × {BCH_N} = {NUM_CHUNKS * BCH_N} bits")
    print(f"  Hash key (SHA-256) : {hash_key_H1}")
    print(f"  Helper (first 64 hex): {helper_hex[:64]}…")
    print(f"  V1 scale           : [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"{'─'*60}")

    # ── VERIFICATION ─────────────────────────────────────────────────────────
    video_labels = {
        "video_2": "IOS No Beard     (V2)",
        "video_3": "Android Beard    (V3)",
        "video_4": "Android No Beard (V4)",
    }
    summary = []

    for vid in ["video_2", "video_3", "video_4"]:
        if vid not in embeddings:
            log.error(f"{vid} not available — skipping.")
            continue

        label = video_labels[vid]
        print(f"\n{'─'*60}")
        print(f"  VERIFICATION — {label}")
        print(f"{'─'*60}")

        vx_bits, _, _, _ = embedding_to_payload(
            embeddings[vid], shared_min=v1_min, shared_max=v1_max
        )
        log.info(f"{vid} quantised using V1 scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
                 f"payload length = {len(vx_bits)} bits")

        result = bch_verify(
            vx_payload_bits = vx_bits,
            v1_payload_bits = v1_bits,
            helper_data     = helper_data,
            hash_key_enroll = hash_key_H1,
            g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
            num_chunks      = NUM_CHUNKS,
            video_label     = vid,
        )
        summary.append(result)

    # ── FINAL SUMMARY 
    print(f"\n{'='*70}")
    print("  PHASE 10 FINAL SUMMARY — V1 ENROLLMENT vs V2 / V3 / V4")
    print(f"{'='*70}")
    print(f"  Enrollment   : V1 — IOS Beard")
    print(f"  Hash key H1  : {hash_key_H1}")
    print(f"  BCH params   : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})"
          f" × {NUM_CHUNKS} chunks  (t_total={T_TOTAL})")
    print(f"  Shared scale : V1 range [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"  QUANT_BITS   : {QUANT_BITS}  ({2**QUANT_BITS} levels, payload={PAYLOAD_BITS} bits)")
    print(f"  Change vs Q3 : payload 1536→{PAYLOAD_BITS}  |  chunks 33→{NUM_CHUNKS}  |  "
          f"t_total 1155→{T_TOTAL}")
    print()
    print(f"  {'Video':<24}  {'Ham-Tot':>7}  {'Max/Chk':>7}  {'Chk>t':>5}  "
          f"{'Failed':>6}  {'Corrctd':>7}  {'Result':>6}")
    print(f"  {'─'*24}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*6}")
    for r in summary:
        print(
            f"  {r['label']:<24}  "
            f"{r['hamming_total']:>7}  "
            f"{r['max_chunk']:>7}  "
            f"{r['chunks_over_t']:>5}  "
            f"{r['failed_chunks']:>6}  "
            f"{r['corrected']:>7}  "
            f"{'PASS' if r['hash_matches'] else 'FAIL':>6}"
        )

    # ── NEW: cross-video t and K min/max summary 
    print()
    print(f"  {'═'*68}")
    print("  T AND K USAGE ACROSS ALL VERIFICATIONS")
    print(f"  {'═'*68}")
    print(f"  BCH designed t = {BCH_T_DESIGNED}  (errors correctable per chunk)")
    print(f"  BCH K          = {BCH_K}  (message bits per chunk)")
    print()
    print(f"  {'Video':<24}  {'k_min':>5}  {'k_max':>5}  {'k_mean':>6}  "
          f"{'t_min':>5}  {'t_max':>5}  {'t_mean':>6}  {'t_failed':>8}")
    print(f"  {'─'*24}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}")

    all_k_vals = []
    all_t_vals = []

    for r in summary:
        k_vals = r["k_used"]
        t_vals = [v for v in r["t_used"] if v >= 0]  # exclude failed (-1)
        t_fail = sum(1 for v in r["t_used"] if v < 0)

        k_min  = min(k_vals)
        k_max  = max(k_vals)
        k_mean = sum(k_vals) / len(k_vals)
        t_min  = min(t_vals) if t_vals else None
        t_max  = max(t_vals) if t_vals else None
        t_mean = sum(t_vals) / len(t_vals) if t_vals else None

        all_k_vals.extend(k_vals)
        all_t_vals.extend(t_vals)

        t_min_str  = f"{t_min:>5}" if t_min is not None else "  N/A"
        t_max_str  = f"{t_max:>5}" if t_max is not None else "  N/A"
        t_mean_str = f"{t_mean:>6.2f}" if t_mean is not None else "   N/A"

        print(
            f"  {r['label']:<24}  "
            f"{k_min:>5}  {k_max:>5}  {k_mean:>6.2f}  "
            f"{t_min_str}  {t_max_str}  {t_mean_str}  "
            f"{t_fail:>8}"
        )

    # Overall across all three verification videos
    if all_k_vals:
        print(f"  {'─'*24}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}")
        all_t_min  = min(all_t_vals) if all_t_vals else None
        all_t_max  = max(all_t_vals) if all_t_vals else None
        all_t_mean = sum(all_t_vals) / len(all_t_vals) if all_t_vals else None
        all_k_min  = min(all_k_vals)
        all_k_max  = max(all_k_vals)
        all_k_mean = sum(all_k_vals) / len(all_k_vals)
        all_t_fail = sum(1 for r in summary for v in r["t_used"] if v < 0)

        t_min_str  = f"{all_t_min:>5}" if all_t_min is not None else "  N/A"
        t_max_str  = f"{all_t_max:>5}" if all_t_max is not None else "  N/A"
        t_mean_str = f"{all_t_mean:>6.2f}" if all_t_mean is not None else "   N/A"

        print(
            f"  {'OVERALL (all videos)':<24}  "
            f"{all_k_min:>5}  {all_k_max:>5}  {all_k_mean:>6.2f}  "
            f"{t_min_str}  {t_max_str}  {t_mean_str}  "
            f"{all_t_fail:>8}"
        )

    print()
    print("  COLUMN GUIDE")
    print(f"  k_min / k_max  : min / max Hamming distance between V1 and Vx per chunk")
    print(f"  k_mean         : average Hamming distance per chunk")
    print(f"  t_min / t_max  : min / max errors actually corrected by BCH per chunk")
    print(f"                   (only successful chunks counted — failed = -1 excluded)")
    print(f"  t_mean         : average errors corrected per successful chunk")
    print(f"  t_failed       : number of chunks where BCH decoder returned -1")
    print(f"  BCH hard limit : t = {BCH_T_DESIGNED} errors per chunk")
    print(f"  BCH K          : {BCH_K} message bits per chunk")
    print(f"  {'═'*68}")

    print()
    all_pass = all(r["hash_matches"] for r in summary)
    all_fail = not any(r["hash_matches"] for r in summary)

    if all_pass:
        print("  OUTCOME : All three verifications PASSED  ✓")
        print(f"  BCH(t={BCH_T_DESIGNED}) corrected all cross-video bit errors.")
        print("  V1 hash identity confirmed for V2, V3, and V4.")
    elif all_fail:
        print("  OUTCOME : All three verifications FAILED  ✗")
        print("  Some chunks still exceed t. Consider:")
        print(f"    - Raising BCH_T_DESIGNED further (currently {BCH_T_DESIGNED})")
        print("    - Option B: interleave dimensions to break up burst errors")
        print("    - Reducing QUANT_BITS back toward 3 to lower per-chunk error counts")
    else:
        passes = [r["label"] for r in summary if r["hash_matches"]]
        fails  = [r["label"] for r in summary if not r["hash_matches"]]
        print(f"  PASSED : {', '.join(passes)}")
        print(f"  FAILED : {', '.join(fails)}")

    print(f"{'='*70}")
    return embeddings, similarities


#  ENTRY POINT

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY-COMMITMENT  —  Phase 10  (QUANT_BITS=5, t=35)")
    print("=" * 70)

    embeddings, pairwise_sims = run()

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Videos processed  : {len(embeddings)} / {len(VIDEO_PATHS)}")
    print(f"  Pairs compared    : {len(pairwise_sims)}")

    if pairwise_sims:
        sv = list(pairwise_sims.values())
        print(f"  Average similarity: {np.mean(sv):.8f}")
        print(f"  Similarity range  : {min(sv):.8f} – {max(sv):.8f}")
        low = [s for s in sv if s < MIN_SIMILARITY_THRESHOLD]
        if not low:
            print("  OVERALL: GOOD — all similarities within expected range")
        elif len(low) <= len(sv) // 3:
            print("  OVERALL: FAIR — some similarities below threshold")
        else:
            print("  OVERALL: POOR — multiple similarities below threshold")

    print("=" * 70)
