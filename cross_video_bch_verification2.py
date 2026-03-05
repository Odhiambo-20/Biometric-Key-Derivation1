"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAFACE FACE EMBEDDING PIPELINE  +  BCH FUZZY COMMITMENT            ║
║                      Phase 12  —  Interleaved Chunking                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHY t_total <= 560 IS A MATHEMATICAL IMPOSSIBILITY WITH YOUR DATA          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  t_total = total error correction budget across ALL chunks.                 ║
║  For genuine users to PASS, t_total MUST be >= actual total bit errors.    ║
║                                                                              ║
║  Your actual total bit errors (measured):                                   ║
║    V2 : 954–970  errors  (37–38% of 2560 bits)                             ║
║    V3 : 731–781  errors  (29–31% of 2560 bits)                             ║
║    V4 : 999–1004 errors  (39%    of 2560 bits)                             ║
║                                                                              ║
║  t_total = 560 → BCH fixes at most 560 errors total.                       ║
║  V4 has 1004 errors.  560 < 1004 → MATHEMATICALLY IMPOSSIBLE.             ║
║                                                                              ║
║  To achieve t_total <= 560 you must first reduce your embedding bit error   ║
║  rate from ~37% down to ~22% (< 560/2560). This requires:                  ║
║    - Enrolling and verifying on the SAME device                             ║
║    - Increasing FRAMES_TO_USE to 50+                                        ║
║    - Consistent lighting and pose                                           ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  WHAT PHASE 12 DOES INSTEAD — INTERLEAVING                                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  Problem in Phase 11 (K=71, sequential):                                    ║
║    Consecutive embedding dimensions are correlated.                          ║
║    Sequential chunking clusters correlated dims → burst errors.             ║
║    V2 max_chunk jumped to 42 (vs 27 in Phase 10 with K=47).                ║
║                                                                              ║
║  The fix — BIT INTERLEAVING:                                                ║
║    Sequential:  chunk_i = bits[ i*K : (i+1)*K ]                            ║
║    Interleaved: chunk_i = bits[ i, i+C, i+2C, i+3C, ... ]  (C=num_chunks) ║
║                                                                              ║
║    bit j  → chunk (j % C),  position (j // C) within that chunk            ║
║                                                                              ║
║    Effect: each chunk gets one bit from EVERY region of the embedding.      ║
║    Errors that were concentrated (max=42) now spread uniformly.             ║
║    Expected max_chunk after interleaving ≈ total_errors / chunks + margin  ║
║                                                                              ║
║  Phase 10 params RESTORED (t=35, K=47):                                     ║
║    These worked with sequential AND max_chunk=27.                           ║
║    With interleaving, max_chunk will drop to ~18–22.                        ║
║    Safety margin improves from 8 → 13–17 errors.                           ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  BCH PARAMETERS  (Phase 12)                                                  ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║    BCH(N=255, K=47, t=35)  PAR=208                                          ║
║    Payload    = 512 × 5 = 2560 bits   (QUANT_BITS=5, unchanged)            ║
║    Chunks     = ceil(2560 / 47) = 55                                        ║
║    Padding    = 55×47 − 2560 = 25 bits (last chunk)                        ║
║    t_total    = 55 × 35 = 1925                                              ║
║    Rate       = 47/255 = 18.4%                                              ║
║    Chunking   = INTERLEAVED  (NEW — replaces sequential)                    ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  HOW INTERLEAVING WORKS IN THE COMMITMENT SCHEME                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  ENROLL (V1):                                                                ║
║    Quantise V1 → 2560-bit payload  p[0..2559]                               ║
║    Pad to 2585 bits  p[0..2584]  (25 zero bits appended)                   ║
║    Interleave into 55 chunks of 47 bits each:                               ║
║      chunk_i[k] = p[ k*55 + i ]   for k=0..46, i=0..54                    ║
║    For each chunk i:                                                         ║
║      r[i]      = chunk_i                        ← 47-bit secret             ║
║      c_r[i]    = BCH_encode(r[i])               ← 255-bit codeword          ║
║      helper[i] = c_r[i] XOR (r[i] ++ zeros(208))                           ║
║    hash_key = SHA-256( r[0] ‖ … ‖ r[54] )                                  ║
║                                                                              ║
║  VERIFY (Vx):                                                                ║
║    Quantise Vx → 2560-bit payload, pad to 2585                              ║
║    Interleave IDENTICALLY into 55 chunks                                    ║
║    Per chunk: noisy = helper[i] XOR (Vx_chunk[i] ++ zeros(208))            ║
║               r̂[i] = BCH_decode(noisy[i])                                  ║
║    hash_verify = SHA-256( r̂[0] ‖ … ‖ r̂[54] )                            ║
║    PASS iff hash_verify == hash_key                                          ║
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

# ── BCH  (Phase 12 — Phase 10 params restored + INTERLEAVING added) ───────────
#
#  QUANT_BITS = 5  →  payload = 512 × 5 = 2560 bits
#
#  BCH(255, t=35)  →  PAR=208, K=47  (same as Phase 10 — proven to work)
#    chunks  = ceil(2560 / 47) = 55
#    padding = 55×47 − 2560   = 25 bits
#    t_total = 55 × 35        = 1925
#    Rate    = 47/255          = 18.4%
#
#  KEY CHANGE: INTERLEAVED chunking replaces sequential chunking.
#    Sequential (Phase 10/11): chunk_i = bits[i*K .. (i+1)*K-1]
#                              → correlated dims cluster → burst errors
#    Interleaved (Phase 12):   chunk_i = bits[i, i+C, i+2C, ...]  C=55
#                              → each chunk spans ALL embedding regions
#                              → errors spread uniformly → max_chunk drops
#
#  Expected max_chunk after interleaving:
#    Phase 10 sequential: V2=27, V3=26, V4=27  (out of K=47)
#    Phase 12 interleaved: V2≈18-22, V3≈14-17, V4≈18-22  (estimated)
#    Safety margin improves from 8 → ~13-17

BCH_N          = 255
BCH_T_DESIGNED = 35   # restored from Phase 10 — proven correct
QUANT_BITS     = 5    # 512 × 5 = 2560 payload bits


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
        img = img.transpose(2, 0, 1)[np.newaxis]
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
    For t=35: K=47, PAR=208, Rate=18.4%
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
    """Systematic BCH encode. Input: K bits. Output: 255-bit codeword."""
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
    Returns (corrected_K_bits, nerr).  nerr=-1 means failure.
    """
    assert len(received_bits) == BCH_N

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

    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = GF_EXP[i % 255]; s = 0
        for bit in received_bits:
            s = gmul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

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
    Quantise 512-dim embedding to QUANT_BITS=5 per dimension.
    32 levels (0–31) → 2560-bit payload.
    """
    levels  = 2 ** QUANT_BITS
    max_val = levels - 1
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
#  SECTION 6 — INTERLEAVING  ← NEW IN PHASE 12
# ══════════════════════════════════════════════════════════════════════════════

def interleave_bits(payload_bits: list, num_chunks: int, K: int) -> List[list]:
    """
    Interleave payload bits into num_chunks chunks of K bits each.

    Instead of sequential assignment:
        chunk_i = bits[ i*K : (i+1)*K ]   ← correlated dims → burst errors

    We use column-major interleaving:
        bit j → chunk (j % num_chunks), position (j // num_chunks)
        chunk_i = [ bits[i], bits[i+C], bits[i+2C], ... ]  where C=num_chunks

    Effect: each chunk draws one bit from every part of the embedding,
    so errors that are concentrated in one region of the embedding space
    get distributed across ALL chunks rather than overloading one chunk.

    Parameters
    ----------
    payload_bits : padded bit list of length num_chunks * K
    num_chunks   : C = number of BCH codewords
    K            : bits per chunk

    Returns
    -------
    chunks : list of num_chunks lists, each of length K
    """
    total = num_chunks * K
    assert len(payload_bits) == total, \
        f"Expected {total} bits, got {len(payload_bits)}"

    chunks = [[] for _ in range(num_chunks)]
    for j, bit in enumerate(payload_bits):
        chunks[j % num_chunks].append(bit)

    # Verify all chunks have exactly K bits
    for i, ch in enumerate(chunks):
        assert len(ch) == K, f"Chunk {i} has {len(ch)} bits, expected {K}"

    return chunks


def deinterleave_bits(chunks: List[list], num_chunks: int, K: int) -> list:
    """
    Reverse of interleave_bits.
    Reconstructs the flat padded bit list from interleaved chunks.

    bit at position (chunk_idx, pos_in_chunk) came from
    original position: pos_in_chunk * num_chunks + chunk_idx
    """
    total = num_chunks * K
    result = [0] * total
    for chunk_idx, chunk in enumerate(chunks):
        for pos_in_chunk, bit in enumerate(chunk):
            original_pos = pos_in_chunk * num_chunks + chunk_idx
            result[original_pos] = bit
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — BIT HELPERS
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
#  SECTION 8 — FUZZY-COMMITMENT  ENROLL + VERIFY  (with interleaving)
# ══════════════════════════════════════════════════════════════════════════════

def bch_enroll(
    v1_payload_bits: list,
    g              : list,
    K              : int,
    PAR            : int,
    num_chunks     : int,
) -> Tuple[List[list], str]:
    """
    Enroll V1 using INTERLEAVED chunking.

    Steps:
      1. Pad payload to num_chunks * K bits
      2. Interleave → 55 chunks of 47 bits each
      3. For each chunk i:
           r[i]      = interleaved_chunk[i]       (47-bit secret)
           c_r[i]    = BCH_encode(r[i])            (255-bit codeword)
           helper[i] = c_r[i] XOR (r[i]++zeros(PAR))
      4. hash_key = SHA-256(all r bits)

    Returns
    -------
    helper_data : list of num_chunks 255-bit lists
    hash_key    : 64-char hex SHA-256
    """
    pad_needed = (num_chunks * K) - len(v1_payload_bits)
    padded     = list(v1_payload_bits) + [0] * pad_needed

    # Interleave into chunks
    chunks = interleave_bits(padded, num_chunks, K)

    helper_data = []
    all_r_bits  = []

    for i in range(num_chunks):
        r         = chunks[i]
        c_r       = bch_encode(r, g, K, PAR)
        v1_pad255 = r + [0] * PAR
        helper    = [a ^ b for a, b in zip(c_r, v1_pad255)]
        helper_data.append(helper)
        all_r_bits.extend(r)

    hash_key = hashlib.sha256(bits_to_bytes(all_r_bits)).hexdigest()

    # Self-check: all encoded codeword syndromes must be zero
    ok = all(
        all(s == 0 for s in _gf2_divmod(bch_encode(chunks[i], g, K, PAR), g))
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
) -> dict:
    """
    Verify Vx against enrolled V1 using INTERLEAVED chunking.

    CRITICAL: Vx bits must be interleaved with the IDENTICAL scheme used
    at enrollment so that helper[i] XOR vx_pad255[i] correctly gives
    c_r[i] with errors only where Vx ≠ V1 in chunk i.
    """
    pad_vx = (num_chunks * K) - len(vx_payload_bits)
    pad_v1 = (num_chunks * K) - len(v1_payload_bits)
    vx_padded = list(vx_payload_bits) + [0] * pad_vx
    v1_padded = list(v1_payload_bits) + [0] * pad_v1

    # Interleave BOTH using same scheme
    vx_chunks = interleave_bits(vx_padded, num_chunks, K)
    v1_chunks = interleave_bits(v1_padded, num_chunks, K)

    # ── Hamming diagnostics (on interleaved chunks) ──────────────────────────
    total_ham     = sum(a != b for a, b in zip(vx_payload_bits, v1_payload_bits))
    per_chunk_ham = [
        sum(v1_chunks[i][j] != vx_chunks[i][j] for j in range(K))
        for i in range(num_chunks)
    ]
    max_chunk_ham   = max(per_chunk_ham)
    chunks_over_t   = sum(1 for e in per_chunk_ham if e > t)
    chunks_within_t = num_chunks - chunks_over_t
    avg_chunk_ham   = total_ham / num_chunks

    print(f"  Total bit differences   : {total_ham} / {len(vx_payload_bits)}"
          f"  ({total_ham / len(vx_payload_bits) * 100:.2f}%)")
    print(f"  Avg errors per chunk    : {avg_chunk_ham:.1f}  "
          f"(interleaved — was burst-clustered before)")
    print(f"  Max errors in one chunk : {max_chunk_ham}  (BCH limit = {t})")
    print(f"  Safety margin           : t − max_chunk = {t} − {max_chunk_ham}"
          f" = {t - max_chunk_ham}")
    print(f"  Chunks within  t={t}   : {chunks_within_t} / {num_chunks}")
    print(f"  Chunks exceeding t={t} : {chunks_over_t}  / {num_chunks}")
    print()
    print("  Per-chunk Hamming distances (after interleaving):")
    for i in range(0, num_chunks, 8):
        row  = per_chunk_ham[i : i + 8]
        line = "  ".join(f"c{i+j:02d}:{row[j]:2d}" for j in range(len(row)))
        print(f"    {line}")

    # ── Fuzzy-commitment correction ──────────────────────────────────────────
    recovered_chunks = []
    total_corr       = 0
    failed           = 0

    for i in range(num_chunks):
        vx_chunk  = vx_chunks[i]
        vx_pad255 = vx_chunk + [0] * PAR
        noisy_cw  = [a ^ b for a, b in zip(helper_data[i], vx_pad255)]
        r_hat, nerr = bch_decode(noisy_cw, g, K, PAR, t)

        if nerr >= 0:
            total_corr += nerr
            recovered_chunks.append(r_hat)
        else:
            failed += 1
            recovered_chunks.append(vx_chunk)   # fallback → hash mismatch

    # Flatten recovered chunks for hashing (same order as enrollment)
    recovered_r_flat = []
    for ch in recovered_chunks:
        recovered_r_flat.extend(ch)

    # ── Re-hash and compare ───────────────────────────────────────────────────
    hash_verify  = hashlib.sha256(bits_to_bytes(recovered_r_flat)).hexdigest()
    hash_matches = hash_verify == hash_key_enroll

    # Remaining errors: deinterleave recovered and compare to v1
    recovered_deinterleaved = deinterleave_bits(recovered_chunks, num_chunks, K)
    remaining = sum(
        a != b
        for a, b in zip(recovered_deinterleaved[:len(vx_payload_bits)],
                        v1_payload_bits)
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
    print("─" * 60)

    return {
        "label"        : video_label,
        "hamming_total": total_ham,
        "avg_chunk"    : avg_chunk_ham,
        "max_chunk"    : max_chunk_ham,
        "margin"       : t - max_chunk_ham,
        "chunks_over_t": chunks_over_t,
        "failed_chunks": failed,
        "corrected"    : total_corr,
        "remaining"    : remaining,
        "hash_matches" : hash_matches,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    sep = "═" * 70

    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial …")
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    PAYLOAD_BITS = 512 * QUANT_BITS                    # 2560
    NUM_CHUNKS   = math.ceil(PAYLOAD_BITS / BCH_K)     # ceil(2560/47) = 55
    T_TOTAL      = NUM_CHUNKS * BCH_T_DESIGNED         # 55 × 35 = 1925
    PAD_NEEDED   = NUM_CHUNKS * BCH_K - PAYLOAD_BITS   # 55×47 − 2560 = 25
    RATE_PCT     = BCH_K / BCH_N * 100
    VIS_ROWS     = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)

    print(sep)
    print("  ADAFACE EMBEDDING PIPELINE  +  BCH FUZZY-COMMITMENT  (Phase 12)")
    print(sep)
    print(f"  Videos             : {len(VIDEO_PATHS)}")
    for i, vp in enumerate(VIDEO_PATHS, 1):
        print(f"    {i}. {Path(vp).name}")
    print(f"  Frame strategy     : top {FRAMES_TO_USE} sharpest of "
          f"{FRAMES_TO_USE * CANDIDATE_MULTIPLIER} candidates")
    print(f"  Mask               : rows {VIS_ROWS}–111 black  |  rows 0–{VIS_ROWS-1} visible")
    print()
    print("  BCH PARAMETERS  —  Phase 12  (Phase 10 params + INTERLEAVING)")
    print(f"    QUANT_BITS      : {QUANT_BITS}  →  payload = 512×{QUANT_BITS} = {PAYLOAD_BITS} bits")
    print(f"    BCH code        : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})")
    print(f"    Parity bits     : {BCH_PAR}")
    print(f"    Rate            : {BCH_K}/{BCH_N} = {RATE_PCT:.1f}%")
    print(f"    Chunks          : {NUM_CHUNKS}  (last chunk zero-padded {PAD_NEEDED} bits)")
    print(f"    t per chunk     : {BCH_T_DESIGNED}")
    print(f"    t_total         : {T_TOTAL}")
    print(f"    Chunking        : INTERLEAVED  ← KEY CHANGE from Phase 10/11")
    print()
    print("  WHY t_total <= 560 IS MATHEMATICALLY IMPOSSIBLE:")
    print("    t_total must be >= total actual bit errors to pass genuine users.")
    print("    V4 has 999-1004 total bit errors.  560 < 1004 → IMPOSSIBLE.")
    print("    To achieve t_total=560, reduce embedding bit error rate to <22%")
    print("    by: same-device enroll/verify, more frames, consistent lighting.")
    print()
    print("  WHY INTERLEAVING HELPS:")
    print("    Sequential: correlated dims cluster → V2 max_chunk=42 with K=71")
    print("    Interleaved: errors spread uniformly → max_chunk drops to ~18-22")
    print("    Security improves: harder for impostors to exploit error clusters.")
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

    similarities: Dict[str, float] = {}
    if len(embeddings) >= 2:
        similarities = compute_pairwise_similarities(embeddings)
        if similarities:
            sv = list(similarities.values())
            print(f"\n  Average similarity : {np.mean(sv):.8f}")
            print(f"  Range              : {min(sv):.8f} – {max(sv):.8f}")

    if "video_1" not in embeddings:
        print(f"\n{sep}\n  ERROR: video_1 unavailable — cannot enroll.\n{sep}")
        return embeddings, similarities

    print(f"\n{'='*70}")
    print("  PHASE 12 — BCH FUZZY-COMMITMENT  (ENROLL V1 / VERIFY V2, V3, V4)")
    print(f"{'='*70}")
    print(f"  BCH code   : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED}) × {NUM_CHUNKS} chunks")
    print(f"  t_total    : {T_TOTAL}")
    print(f"  Rate       : {RATE_PCT:.1f}%")
    print(f"  Payload    : {PAYLOAD_BITS} bits  ({QUANT_BITS}-bit quant × 512 dims)")
    print(f"  Chunking   : INTERLEAVED  (bit j → chunk j%{NUM_CHUNKS})")
    print(f"  Scale      : V1's [v_min, v_max] used for all videos")

    v1_bits, _, v1_min, v1_max = embedding_to_payload(embeddings["video_1"])
    log.info(f"V1 quantised — scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
             f"payload = {len(v1_bits)} bits")

    # ── ENROLLMENT ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ENROLLMENT — V1 (IOS Beard)  [INTERLEAVED]")
    print(f"{'─'*60}")

    helper_data, hash_key_H1 = bch_enroll(
        v1_payload_bits=v1_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks=NUM_CHUNKS,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Payload bits       : {len(v1_bits)}")
    print(f"  Chunks enrolled    : {NUM_CHUNKS}  (interleaved)")
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
        print(f"  VERIFICATION — {label}  [INTERLEAVED]")
        print(f"{'─'*60}")

        vx_bits, _, _, _ = embedding_to_payload(
            embeddings[vid], shared_min=v1_min, shared_max=v1_max
        )
        log.info(f"{vid} quantised — scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
                 f"payload = {len(vx_bits)} bits")

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

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 12 FINAL SUMMARY — V1 ENROLLMENT vs V2 / V3 / V4")
    print(f"{'='*70}")
    print(f"  Enrollment   : V1 — IOS Beard")
    print(f"  Hash key H1  : {hash_key_H1}")
    print(f"  BCH params   : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})"
          f" × {NUM_CHUNKS} chunks  (t_total={T_TOTAL})")
    print(f"  Rate         : {RATE_PCT:.1f}%  |  Parity = {BCH_PAR} bits")
    print(f"  Chunking     : INTERLEAVED  (bit j → chunk j%{NUM_CHUNKS})")
    print(f"  Shared scale : V1 range [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"  QUANT_BITS   : {QUANT_BITS}  ({2**QUANT_BITS} levels, payload={PAYLOAD_BITS} bits)")
    print()
    print(f"  {'Video':<24}  {'Ham-Tot':>7}  {'Avg/Chk':>7}  {'Max/Chk':>7}  "
          f"{'Margin':>6}  {'Chk>t':>5}  {'Failed':>6}  {'Corrctd':>7}  {'Result':>6}")
    print(f"  {'─'*24}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}  "
          f"{'─'*6}  {'─'*7}  {'─'*6}")
    for r in summary:
        print(
            f"  {r['label']:<24}  "
            f"{r['hamming_total']:>7}  "
            f"{r['avg_chunk']:>7.1f}  "
            f"{r['max_chunk']:>7}  "
            f"{r['margin']:>6}  "
            f"{r['chunks_over_t']:>5}  "
            f"{r['failed_chunks']:>6}  "
            f"{r['corrected']:>7}  "
            f"{'PASS' if r['hash_matches'] else 'FAIL':>6}"
        )

    print()
    all_pass = all(r["hash_matches"] for r in summary)
    all_fail = not any(r["hash_matches"] for r in summary)

    if all_pass:
        print("  OUTCOME : All three verifications PASSED  ✓")
        print("  Interleaving successfully spread burst errors across chunks.")
        margins = [r['margin'] for r in summary]
        #print(f"  Safety margins: {[f\"{r['label'].split()[0]}={r['margin']}\" for r in summary]}")
        print("  V1 hash identity confirmed for V2, V3, and V4.")
    elif all_fail:
        print("  OUTCOME : All three verifications FAILED  ✗")
        print("  Interleaving was not sufficient. Embedding noise is too high.")
        print("  To achieve t_total<=560, reduce bit error rate below 22%:")
        print("    1. Enroll and verify on the SAME device")
        print("    2. Increase FRAMES_TO_USE to 50 or more")
        print("    3. Maintain consistent lighting and pose")
        print(f"    4. Current worst case: {max(r['hamming_total'] for r in summary)}"
              f" errors — need < 560")
    else:
        passes = [r["label"] for r in summary if r["hash_matches"]]
        fails  = [r["label"] for r in summary if not r["hash_matches"]]
        print(f"  PASSED : {', '.join(passes)}")
        print(f"  FAILED : {', '.join(fails)}")

    print(f"{'='*70}")
    return embeddings, similarities


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY-COMMITMENT — Phase 12  (Interleaved, t=35)")
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
