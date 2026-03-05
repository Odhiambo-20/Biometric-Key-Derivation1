"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAFACE FACE EMBEDDING PIPELINE  +  BCH FUZZY COMMITMENT            ║
║              Phase 12 — BCH ONLY (No Identity Gates G1/G2)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSTRUCTION 1:                                                              ║
║    V1 runs BCH enroll -> creates helper_data + HMAC-SHA256 commit           ║
║    (256-bit hash key using random 256-bit SALT)                             ║
║                                                                              ║
║  INSTRUCTION 2:                                                              ║
║    V2,V3,V4,V5,V6,V7 each run BCH verify separately using V1 helper_data   ║
║    -> attempt to recover V1's secret r -> recompute HMAC -> match or not   ║
║                                                                              ║
║  SINGLE GATE: Gate 3 only — BCH decode all chunks + HMAC-SHA256 match      ║
║                                                                              ║
║  BCH PARAMETERS:                                                             ║
║  QUANT_BITS = 4  ->  payload = 512 x 4 = 2048 bits                        ║
║  BCH(N=255, t=35)  ->  PAR=208, K=47                                       ║
║  Chunks = ceil(2048/47) = 44,  t_total = 44 x 35 = 1540                   ║
║  PAD    = 44 x 47 - 2048 = 20 bits (zero-padded)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import math
import os
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
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",
    "/home/victor/Documents/Desktop/Embeddings/IOS -Sha V6 .MOV",
    "/home/victor/Documents/Desktop/Embeddings/IOS - Rusl V7.mov",
]
WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)
OUTPUT_ROOT = "masked_frames"

FRAMES_TO_USE        = 20
CANDIDATE_MULTIPLIER = 3
FACE_SIZE            = 112
MASK_FRACTION        = 0.38

# ── BCH parameters ───────────────────────────────────────────────────────────
BCH_N          = 255
BCH_T_DESIGNED = 35
QUANT_BITS     = 4


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
    log.info(f"  Scanning {n_candidates} candidates -> top {num_frames} by sharpness")
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
            log.info(f"  Frame {pos:>5}: face {crop.shape[1]}x{crop.shape[0]}px")
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
    log.info(f"  Masked photo saved -> {save_path}")
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
    print(f"    Visible rows  0-{visible_rows-1:<2}           : forehead, eyes, nose, nostrils")
    print(f"    Black   rows  {visible_rows}-111           : mouth, chin")
    print(f"  Saved photo                     : {save_path}")
    print_embedding(final, video_name)
    return video_name, final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COSINE SIMILARITY (kept for reference/logging only)
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    sim = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if label:
        log.info(f"Cosine similarity [{label}] : {sim:.8f}")
    return sim


def compute_pairwise_similarities(
    emb_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    similarities = {}
    print("\n" + "=" * 62)
    print("  PAIRWISE COSINE SIMILARITY COMPARISONS  (informational only)")
    print("=" * 62)
    for v1n, v2n in itertools.combinations(emb_dict.keys(), 2):
        e1, e2 = emb_dict[v1n], emb_dict[v2n]
        sim    = cosine_similarity(e1, e2, label=f"{v1n}_vs_{v2n}")
        key    = f"{v1n}_vs_{v2n}"
        similarities[key] = sim
        print(f"  {v1n} vs {v2n} : {sim:.8f}")
    return similarities


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GF(2) / GF(2^8) ARITHMETIC
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
    assert len(msg_bits) == K, f"Expected {K} bits, got {len(msg_bits)}"
    padded    = list(msg_bits) + [0] * PAR
    remainder = _gf2_divmod(padded, g)
    parity    = _poly_pad(remainder, PAR)
    return list(msg_bits) + parity


def bch_decode(
    received_bits: list,
    g: list, K: int, PAR: int, t: int,
) -> Tuple[list, int]:
    assert len(received_bits) == BCH_N, \
        f"Expected {BCH_N} bits, got {len(received_bits)}"

    GF_EXP = [0] * 512; GF_LOG = [0] * 256; x = 1
    for i in range(255):
        GF_EXP[i] = GF_EXP[i + 255] = x
        GF_LOG[x] = i
        x = _gf256_mul(x, 2)

    def gmul(a, b):
        return 0 if (a == 0 or b == 0) \
            else GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255]
    def ginv(a):
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
            T = list(C)
            coef = gmul(d, ginv(b))
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
    emb: np.ndarray,
    shared_min: Optional[float] = None,
    shared_max: Optional[float] = None,
) -> Tuple[list, np.ndarray, float, float]:
    levels  = 2 ** QUANT_BITS
    max_val = levels - 1
    v_min   = shared_min if shared_min is not None else float(emb.min())
    v_max   = shared_max if shared_max is not None else float(emb.max())
    q_vec   = np.clip(
        np.round((emb - v_min) / (v_max - v_min) * max_val), 0, max_val
    ).astype(np.int32)
    bits = []
    for q in q_vec:
        for shift in range(QUANT_BITS - 1, -1, -1):
            bits.append(int((int(q) >> shift) & 1))
    return bits, q_vec, v_min, v_max


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BIT / BYTE HELPERS
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
        format(b[i] * 8 + b[i+1] * 4 + b[i+2] * 2 + b[i+3], "x")
        for i in range(0, len(b), 4)
    )


def hmac_commit(salt: bytes, message_bits: list) -> str:
    """HMAC-SHA256 with 256-bit random salt -> 256-bit (64 hex char) hash key."""
    return hmac.new(salt, bits_to_bytes(message_bits), hashlib.sha256).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — ENROLL  (Instruction 1)
#
#  V1 runs BCH:
#    - generate random 256-bit SALT
#    - for each chunk: r[i] = V1_chunk[i]
#                      cw_r[i] = BCH_encode(r[i])
#                      helper[i] = cw_r[i] XOR (r[i] ++ zeros(PAR))
#    - commit = HMAC-SHA256(SALT, r[0]||r[1]||...||r[N-1])   <- 256-bit hash key
#    - store: helper_data, commit (256-bit hash key), SALT
# ══════════════════════════════════════════════════════════════════════════════

def bch_enroll(
    v1_payload_bits: list,
    g: list, K: int, PAR: int,
    num_chunks: int,
) -> Tuple[List[list], str, bytes]:
    """
    Enroll V1.
    Returns:
        helper_data  : list of num_chunks x 255-bit lists
        commit       : HMAC-SHA256 hex string (256-bit hash key)
        salt         : 32 random bytes (256-bit salt)
    """
    # --- Instruction 1: generate 256-bit random salt ---
    salt   = os.urandom(32)          # 256-bit random salt
    pad    = (num_chunks * K) - len(v1_payload_bits)
    v1_pad = list(v1_payload_bits) + [0] * pad

    helper_data = []
    all_r_bits  = []

    for i in range(num_chunks):
        r         = v1_pad[i * K : (i + 1) * K]   # K-bit secret chunk
        cw_r      = bch_encode(r, g, K, PAR)        # 255-bit BCH codeword
        v1_pad255 = r + [0] * PAR                   # r padded to 255 bits
        helper    = [a ^ b for a, b in zip(cw_r, v1_pad255)]  # XOR sketch
        helper_data.append(helper)
        all_r_bits.extend(r)

    # 256-bit hash key = HMAC-SHA256(SALT, all secret bits)
    commit = hmac_commit(salt, all_r_bits)

    # Sanity: all enrolled codewords have zero syndrome
    ok = all(
        all(s == 0 for s in _gf2_divmod(
            bch_encode(v1_pad[i * K : (i + 1) * K], g, K, PAR), g
        ))
        for i in range(num_chunks)
    )
    log.info(f"Enrollment — all codeword syndromes zero: {ok}  <- must be True")
    log.info(f"Enrollment — SALT (256-bit): {salt.hex()}")
    log.info(f"Enrollment — Commit (256-bit HMAC key): {commit}")

    return helper_data, commit, salt


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — VERIFY  (Instruction 2)
#
#  Each of V2..V7 runs separately using V1's helper_data:
#    noisy[i]  = helper[i] XOR (Vx_chunk[i] ++ zeros(PAR))
#    r_hat[i], nerr = BCH_decode(noisy[i])
#    if any nerr == -1 -> FAIL (chunk uncorrectable)
#    commit_v  = HMAC-SHA256(SALT, r_hat[0]||...||r_hat[N-1])
#    PASS iff all nerr >= 0  AND  commit_v == commit_enroll
# ══════════════════════════════════════════════════════════════════════════════

def bch_verify(
    vx_payload_bits : list,
    v1_payload_bits : list,
    helper_data     : List[list],
    commit_enroll   : str,
    salt            : bytes,
    g               : list,
    K               : int,
    PAR             : int,
    t               : int,
    num_chunks      : int,
    video_label     : str,
) -> dict:
    sep60 = "─" * 60

    # ── Pad payloads to full chunk boundary ───────────────────────────────────
    pad_vx = (num_chunks * K) - len(vx_payload_bits)
    pad_v1 = (num_chunks * K) - len(v1_payload_bits)
    vx_ext = list(vx_payload_bits) + [0] * pad_vx
    v1_ext = list(v1_payload_bits) + [0] * pad_v1

    # ── Payload-level Hamming (informational, not a gate) ─────────────────────
    total_ham = sum(a != b for a, b in zip(vx_payload_bits, v1_payload_bits))
    ham_rate  = total_ham / len(v1_payload_bits)

    # ── Per-chunk Hamming distances (informational) ───────────────────────────
    per_chunk_ham = [
        sum(v1_ext[i*K+j] != vx_ext[i*K+j] for j in range(K))
        for i in range(num_chunks)
    ]
    max_chunk_ham   = max(per_chunk_ham)
    chunks_over_t   = sum(1 for e in per_chunk_ham if e > t)
    chunks_within_t = num_chunks - chunks_over_t

    print(f"  Total bit differences    : {total_ham} / {len(vx_payload_bits)}"
          f"  ({ham_rate * 100:.2f}%)  [informational only]")
    print(f"  Max errors in one chunk  : {max_chunk_ham}  (BCH limit t = {t})")
    print(f"  Chunks within  t={t}   : {chunks_within_t:>3} / {num_chunks}"
          f"  {'<-- all chunks correctable ✓' if chunks_within_t == num_chunks else ''}")
    print(f"  Chunks exceeding t={t}  : {chunks_over_t:>3}  / {num_chunks}"
          f"  {'<-- BCH will FAIL these chunks' if chunks_over_t > 0 else ''}")
    print()
    print("  Per-chunk Hamming distances:")
    for i in range(0, num_chunks, 8):
        row  = per_chunk_ham[i : i + 8]
        line = "  ".join(f"c{i+j:02d}:{row[j]:2d}" for j in range(len(row)))
        print(f"    {line}")

    # ── BCH decode each chunk using V1 helper_data ────────────────────────────
    recovered_r      = []
    total_corr       = 0
    bch_failed_count = 0
    t_used           = []
    k_used           = []

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
            bch_failed_count += 1
            recovered_r.extend(vx_chunk)   # keep noisy bits (HMAC will mismatch)

    # ── GATE 3: BCH success + HMAC commitment check ───────────────────────────
    commit_verify  = hmac_commit(salt, recovered_r)
    bch_hmac_match = commit_verify == commit_enroll
    gate3_pass     = (bch_failed_count == 0) and bch_hmac_match

    remaining = sum(
        a != b for a, b in zip(recovered_r, v1_ext[:num_chunks * K])
    )

    print()
    print(f"  BCH errors corrected     : {total_corr}")
    print(f"  Failed BCH chunks        : {bch_failed_count}"
          f"  {'(0 = full recovery ✓)' if bch_failed_count == 0 else '(<- nonzero -> FAIL)'}")
    print(f"  Remaining bit errors     : {remaining}"
          f"  {'(0 = perfect recovery ✓)' if remaining == 0 else ''}")
    print()
    print(f"  Commit V1 enroll  (256-bit HMAC key) : {commit_enroll}")
    print(f"  Commit {video_label:>7} result (256-bit HMAC key) : {commit_verify}")
    print(f"  HMAC match               : {'YES ✓' if bch_hmac_match else 'NO ✗'}")
    print()
    print(f"  [Gate 3] BCH + HMAC-SHA256 : "
          f"{'PASS ✓' if gate3_pass else 'FAIL ✗'}"
          f"{'  (BCH chunk failures present)' if bch_failed_count > 0 else ''}")

    # ── FINAL VERDICT (single gate: Gate 3 only) ──────────────────────────────
    print()
    if gate3_pass:
        verdict = "PASS  ✓  SAME PERSON — BCH recovered V1 secret, HMAC verified"
    else:
        reasons = []
        if bch_failed_count > 0:
            reasons.append(
                f"BCH FAIL ({bch_failed_count} chunks uncorrectable, errors > t={t})"
            )
        if not bch_hmac_match:
            reasons.append("HMAC MISMATCH — recovered bits != V1 secret")
        verdict = "FAIL  ✗  REJECTED — " + " | ".join(reasons)

    print(f"  Final Verdict            : {verdict}")

    # ── Per-chunk detail ──────────────────────────────────────────────────────
    successful_t = [v for v in t_used if v >= 0]
    failed_t     = [i for i, v in enumerate(t_used) if v < 0]

    print()
    print("  ── PER-CHUNK t / K USAGE ──────────────────────────────────")
    print(f"  BCH t={t}  K={K}  N={BCH_N}")
    print()
    print("  chunk | k_used (payload Hamming) | t_used (BCH corrected)")
    print("  " + "─" * 55)
    for i in range(0, num_chunks, 8):
        k_row  = k_used[i : i + 8]
        t_row  = t_used[i : i + 8]
        k_line = "  ".join(f"c{i+j:02d}:{k_row[j]:2d}" for j in range(len(k_row)))
        t_line = "  ".join(
            f"c{i+j:02d}:{'FAIL' if t_row[j] < 0 else str(t_row[j]):>4}"
            for j in range(len(t_row))
        )
        print(f"    k: {k_line}")
        print(f"    t: {t_line}")
        print()

    if successful_t:
        print(f"  t_used (successful chunks only):")
        print(f"    min  : {min(successful_t)}  "
              f"(chunk {t_used.index(min(successful_t)):02d})")
        print(f"    max  : {max(successful_t)}  "
              f"(chunk {t_used.index(max(successful_t)):02d})")
        print(f"    mean : {sum(successful_t)/len(successful_t):.2f}")

    print()
    print(f"  k_used: min={min(k_used)}  max={max(k_used)}  "
          f"mean={sum(k_used)/len(k_used):.2f}")

    if failed_t:
        print(f"\n  FAILED BCH chunks (nerr = -1)  <- errors exceeded t={t}:")
        for ci in failed_t:
            print(f"    chunk {ci:02d}: k_used={k_used[ci]} > t={t}")
    else:
        print(f"\n  All {num_chunks} BCH chunks decoded successfully ✓")
        if gate3_pass:
            print(f"  HMAC-SHA256 verified ✓  ->  256-bit hash keys match -> PASS")

    print("  " + sep60)
    print(sep60)

    return {
        "label"          : video_label,
        "hamming_total"  : total_ham,
        "hamming_rate"   : ham_rate,
        "max_chunk"      : max_chunk_ham,
        "chunks_over_t"  : chunks_over_t,
        "bch_failed"     : bch_failed_count,
        "corrected"      : total_corr,
        "remaining"      : remaining,
        "bch_hmac_match" : bch_hmac_match,
        "gate3_pass"     : gate3_pass,
        "t_used"         : t_used,
        "k_used"         : k_used,
        "t_min"          : min(successful_t) if successful_t else None,
        "t_max"          : max(successful_t) if successful_t else None,
        "k_min"          : min(k_used),
        "k_max"          : max(k_used),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    sep70 = "=" * 70

    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial ...")
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    # ── Derived parameters ────────────────────────────────────────────────────
    PAYLOAD_BITS = 512 * QUANT_BITS
    NUM_CHUNKS   = math.ceil(PAYLOAD_BITS / BCH_K)
    T_TOTAL      = NUM_CHUNKS * BCH_T_DESIGNED
    PAD_NEEDED   = NUM_CHUNKS * BCH_K - PAYLOAD_BITS

    print(sep70)
    print("  ADAFACE + BCH FUZZY COMMITMENT — Phase 12 (BCH Only, No G1/G2)")
    print(sep70)
    print()
    print("  DESIGN (Instructions 1 & 2):")
    print()
    print("  INSTRUCTION 1 — ENROLL V1:")
    print(f"    SALT      = os.urandom(32)                  [256-bit random salt]")
    print(f"    r[i]      = V1_chunk[i]                     (K={BCH_K}-bit secret)")
    print(f"    cw_r[i]   = BCH_encode(r[i])                (255-bit codeword)")
    print(f"    helper[i] = cw_r[i] XOR (r[i]++zeros({BCH_PAR}))   (XOR sketch)")
    print(f"    commit    = HMAC-SHA256(SALT, r[0]||...||r[N])  [256-bit hash key]")
    print(f"    store:      helper_data, commit, SALT")
    print()
    print("  INSTRUCTION 2 — VERIFY V2..V7 (each separately, using V1 helper_data):")
    print(f"    noisy[i]       = helper[i] XOR (Vx_chunk[i]++zeros({BCH_PAR}))")
    print(f"    r_hat[i], nerr = BCH_decode(noisy[i])")
    print(f"    if any nerr == -1  ->  FAIL  (chunk uncorrectable)")
    print(f"    commit_v       = HMAC-SHA256(SALT, r_hat[0]||...||r_hat[N])")
    print(f"    PASS iff  all_nerr >= 0  AND  commit_v == commit")
    print(f"    -> 256-bit hash keys match  =>  SAME PERSON")
    print(f"    -> 256-bit hash keys differ =>  REJECTED")
    print()
    print(f"  BCH PARAMETERS (runtime-computed from t={BCH_T_DESIGNED}):")
    print(f"    BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})   PAR={BCH_PAR}")
    print(f"    Payload  : {PAYLOAD_BITS} bits  (512 dims x {QUANT_BITS} bits/dim)")
    print(f"    Chunks   : {NUM_CHUNKS}  (zero-padded last chunk: {PAD_NEEDED} bits)")
    print(f"    t_total  : {T_TOTAL}  ({NUM_CHUNKS} chunks x t={BCH_T_DESIGNED})")
    print(f"    Helper   : {NUM_CHUNKS} x {BCH_N} = {NUM_CHUNKS * BCH_N} bits stored")
    print(sep70)

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

    print(f"\n{sep70}")
    print("  PROCESSING COMPLETE — FINAL EMBEDDING NORMS")
    print(sep70)
    for name, emb in embeddings.items():
        print(f"  {name:10}: norm = {np.linalg.norm(emb):.8f}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep70)

    # Cosine similarities — informational only, NOT used as a gate
    similarities: Dict[str, float] = {}
    if len(embeddings) >= 2:
        similarities = compute_pairwise_similarities(embeddings)

    if "video_1" not in embeddings:
        print(f"\n{sep70}\n  ERROR: video_1 unavailable — cannot enroll.\n{sep70}")
        return embeddings, similarities

    print(f"\n{'='*70}")
    print("  PHASE 12 — BCH FUZZY COMMITMENT (SINGLE GATE: BCH + HMAC-SHA256)")
    print("  INSTRUCTION 1: ENROLL V1")
    print("  INSTRUCTION 2: VERIFY V2, V3, V4, V5, V6, V7  (each separately)")
    print(f"{'='*70}")
    print(f"  BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED}) x {NUM_CHUNKS} chunks"
          f"  |  QUANT_BITS={QUANT_BITS}  |  t_total={T_TOTAL}")
    print(f"  Quantisation scale: V1's [v_min, v_max] fixed reference for ALL videos")

    # ── Quantise V1 and capture its scale ────────────────────────────────────
    v1_bits, _, v1_min, v1_max = embedding_to_payload(embeddings["video_1"])
    log.info(f"V1 quantised — scale [{v1_min:.5f}, {v1_max:.5f}]"
             f"  |  payload = {len(v1_bits)} bits")

    # ── INSTRUCTION 1: ENROLL V1 ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  INSTRUCTION 1 — ENROLLMENT — V1 (IOS Beard)")
    print(f"{'─'*60}")

    helper_data, commit_H1, enrollment_salt = bch_enroll(
        v1_payload_bits=v1_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks=NUM_CHUNKS,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Payload bits              : {len(v1_bits)}")
    print(f"  Chunks enrolled           : {NUM_CHUNKS}")
    print(f"  Helper size               : {NUM_CHUNKS} x {BCH_N} = {NUM_CHUNKS * BCH_N} bits")
    print(f"  SALT (256-bit, hex)       : {enrollment_salt.hex()}")
    print(f"  Commit (256-bit HMAC key) : {commit_H1}")
    print(f"  Helper (first 64 hex)     : {helper_hex[:64]}...")
    print(f"  V1 scale                  : [{v1_min:.6f}, {v1_max:.6f}]")
    print(f"{'─'*60}")

    # ── VIDEO LABELS ──────────────────────────────────────────────────────────
    video_labels = {
        "video_2": "IOS No Beard     (V2)",
        "video_3": "Android Beard    (V3)",
        "video_4": "Android No Beard (V4)",
        "video_5": "Android Video 5  (V5)",
        "video_6": "IOS Sha          (V6)",
        "video_7": "IOS Rusl         (V7)",
    }
    summary = []

    # ── INSTRUCTION 2: VERIFY V2..V7 (each separately) ───────────────────────
    for vid in ["video_2", "video_3", "video_4", "video_5", "video_6", "video_7"]:
        if vid not in embeddings:
            log.error(f"{vid} not available — skipping.")
            continue

        label = video_labels[vid]
        print(f"\n{'─'*60}")
        print(f"  INSTRUCTION 2 — VERIFICATION — {label}")
        print(f"  Using V1 helper_data + V1 SALT + V1 256-bit commit")
        print(f"{'─'*60}")

        # Quantise Vx using V1's fixed scale
        vx_bits, _, _, _ = embedding_to_payload(
            embeddings[vid], shared_min=v1_min, shared_max=v1_max
        )
        log.info(f"{vid} quantised using V1 scale — payload = {len(vx_bits)} bits")

        result = bch_verify(
            vx_payload_bits=vx_bits,
            v1_payload_bits=v1_bits,
            helper_data    =helper_data,
            commit_enroll  =commit_H1,
            salt           =enrollment_salt,
            g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
            num_chunks     =NUM_CHUNKS,
            video_label    =vid,
        )
        summary.append(result)

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 12 — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Enrolled  : V1 — IOS Beard")
    print(f"  SALT      : {enrollment_salt.hex()[:32]}...  (256-bit)")
    print(f"  Commit H1 : {commit_H1}  (256-bit HMAC-SHA256 key)")
    print(f"  BCH params: BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})"
          f" x {NUM_CHUNKS} chunks  |  t_total={T_TOTAL}")
    print(f"  Gate      : G3 only — BCH(all chunks decoded) + HMAC-SHA256 match")
    print()
    print(f"  {'Video':<24}  {'Ham%':>5}  {'BCHfail':>7}  {'HMAC':>5}"
          f"  {'G3':>4}  {'RESULT':>6}  Note")
    print(f"  {'─'*24}  {'─'*5}  {'─'*7}  {'─'*5}"
          f"  {'─'*4}  {'─'*6}  {'─'*38}")

    for r in summary:
        g3s   = "PASS" if r["gate3_pass"] else "FAIL"
        hmacs = "MATCH" if r["bch_hmac_match"] else "MISS"
        ress  = "PASS ✓" if r["gate3_pass"] else "FAIL ✗"
        hams  = f"{r['hamming_rate']*100:.1f}%"
        bchfs = str(r["bch_failed"])

        if r["gate3_pass"]:
            note = "BCH corrected all chunks, 256-bit HMAC keys match ✓"
        elif r["bch_failed"] > 0:
            note = f"BCH {r['bch_failed']} chunks exceeded t={BCH_T_DESIGNED}"
        else:
            note = "BCH decoded but 256-bit HMAC keys differ ✗"

        print(
            f"  {r['label']:<24}  "
            f"{hams:>5}  {bchfs:>7}  {hmacs:>5}  "
            f"{g3s:>4}  {ress:>6}  {note}"
        )

    # ── T/K statistics ────────────────────────────────────────────────────────
    print()
    print(f"  {'='*68}")
    print("  T AND K USAGE — ALL VIDEOS")
    print(f"  {'='*68}")
    print(f"  BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})  PAR={BCH_PAR}")
    print(f"  QUANT_BITS={QUANT_BITS}  PAYLOAD={PAYLOAD_BITS}"
          f"  CHUNKS={NUM_CHUNKS}  T_TOTAL={T_TOTAL}")
    print()
    print(f"  {'Video':<24}  {'k_min':>5}  {'k_max':>5}  {'k_mean':>6}"
          f"  {'t_min':>5}  {'t_max':>5}  {'t_mean':>6}  {'BCHfail':>7}  Result")
    print(f"  {'─'*24}  {'─'*5}  {'─'*5}  {'─'*6}"
          f"  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*10}")

    all_k = []; all_t = []
    for r in summary:
        k_vals   = r["k_used"]
        t_vals   = [v for v in r["t_used"] if v >= 0]
        all_k.extend(k_vals)
        all_t.extend(t_vals)
        result_s = "PASS ✓" if r["gate3_pass"] else "FAIL ✗"
        t_min_s  = f"{min(t_vals):>5}" if t_vals else "  N/A"
        t_max_s  = f"{max(t_vals):>5}" if t_vals else "  N/A"
        t_mea_s  = f"{sum(t_vals)/len(t_vals):>6.2f}" if t_vals else "   N/A"
        print(
            f"  {r['label']:<24}  "
            f"{min(k_vals):>5}  {max(k_vals):>5}  {sum(k_vals)/len(k_vals):>6.2f}  "
            f"{t_min_s}  {t_max_s}  {t_mea_s}  "
            f"{r['bch_failed']:>7}  {result_s}"
        )

    if all_k:
        print(f"  {'─'*24}  {'─'*5}  {'─'*5}  {'─'*6}"
              f"  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}")
        t_min_s      = f"{min(all_t):>5}" if all_t else "  N/A"
        t_max_s      = f"{max(all_t):>5}" if all_t else "  N/A"
        t_mea_s      = f"{sum(all_t)/len(all_t):>6.2f}" if all_t else "   N/A"
        all_bch_fail = sum(r["bch_failed"] for r in summary)
        print(
            f"  {'OVERALL':<24}  "
            f"{min(all_k):>5}  {max(all_k):>5}  {sum(all_k)/len(all_k):>6.2f}  "
            f"{t_min_s}  {t_max_s}  {t_mea_s}  "
            f"{all_bch_fail:>7}"
        )

    print()
    print("  COLUMN GUIDE")
    print(f"  k_min/k_max/k_mean : per-chunk payload Hamming (K={BCH_K} bits/chunk)")
    print(f"  t_min/t_max/t_mean : errors corrected by BCH (successful chunks only)")
    print(f"  BCHfail            : chunks where errors > t={BCH_T_DESIGNED} (nerr = -1)")
    print(f"  Result             : PASS = BCH decoded all chunks + HMAC keys match")
    print(f"  {'='*68}")

    # ── Outcome summary ───────────────────────────────────────────────────────
    print()
    passed = [r for r in summary if r["gate3_pass"]]
    failed = [r for r in summary if not r["gate3_pass"]]

    print("  OUTCOME (BCH + 256-bit HMAC-SHA256 only):")
    for r in passed:
        print(f"    PASS ✓ : {r['label']}"
              f"  (Ham={r['hamming_rate']*100:.1f}%,"
              f" BCHfail=0,"
              f" remaining_errors={r['remaining']},"
              f" 256-bit HMAC=MATCH)")
    for r in failed:
        reasons = []
        if r["bch_failed"] > 0:
            reasons.append(f"BCHfail={r['bch_failed']} chunks")
        if not r["bch_hmac_match"]:
            reasons.append("256-bit HMAC MISMATCH")
        print(f"    FAIL ✗ : {r['label']}  ({', '.join(reasons)})")

    print()
    print("  CRYPTOGRAPHIC SUMMARY:")
    print(f"    Commitment : HMAC-SHA256 with 256-bit random SALT")
    print(f"                 -> 256-bit hash key stored at enroll")
    print(f"                 -> 256-bit hash key recomputed at verify")
    print(f"                 -> MATCH = same person  |  MISMATCH = rejected")
    print(f"    BCH sketch : helper[i] = BCH_encode(r[i]) XOR (r[i]++zeros({BCH_PAR}))")
    print(f"    t=35, K=47 : genuine per-chunk errors (max ~29) < t=35")
    print(f"                 -> BCH successfully recovers V1 secret for genuine matches")
    print(f"    Parameters : QUANT_BITS={QUANT_BITS}"
          f" | BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})")
    print(f"                 Chunks={NUM_CHUNKS}"
          f" | PAD={PAD_NEEDED} bits | t_total={T_TOTAL}")
    print(f"{'='*70}")

    return embeddings, similarities


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY COMMITMENT — Phase 12 (BCH Only)")
    print("  Single Gate: G3 = BCH(all chunks) + HMAC-SHA256 (256-bit hash key)")
    print("  QUANT_BITS=4  |  BCH(255, K=47, t=35)  |  chunks=44  |  t_total=1540")
    print("  INSTRUCTION 1: V1 enrolls -> helper_data + 256-bit commit")
    print("  INSTRUCTION 2: V2..V7 each verify against V1 helper_data")
    print("=" * 70)

    embeddings, pairwise_sims = run()

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Videos processed  : {len(embeddings)} / {len(VIDEO_PATHS)}")
    print(f"  Pairs compared    : {len(pairwise_sims)}")
    print("=" * 70)
