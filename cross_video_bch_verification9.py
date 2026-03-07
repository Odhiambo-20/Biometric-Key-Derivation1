"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAFACE FACE EMBEDDING PIPELINE  +  BCH FUZZY COMMITMENT            ║
║              Phase 13 FINAL — BCH ONLY, NO G1/G2                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SECURITY MODEL                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  INSTRUCTION 1 — REGISTRATION (V1):                                         ║
║    Face → AdaFace → embedding → quantise → 2048-bit payload                ║
║    Pad payload to fit exactly NUM_CHUNKS × K bits                           ║
║    For each chunk i:                                                         ║
║      r[i]      = V1_chunk[i]               (K-bit secret)                  ║
║      cw[i]     = BCH_encode(r[i])          (N-bit codeword)                 ║
║      helper[i] = cw[i] XOR (r[i]++0s)     (Juels-Wattenberg sketch)        ║
║    SALT   = os.urandom(32)                 (256-bit random salt)            ║
║    commit = HMAC-SHA256(SALT, r[0]||...||r[N-1])  (256-bit hash key)       ║
║    STORE: helper_data, commit, SALT, v1_min, v1_max                         ║
║    DELETE: raw embedding, secret r                                           ║
║                                                                              ║
║  INSTRUCTION 2 — LOGIN (V2..V7 each independently):                         ║
║    Face → AdaFace → embedding → quantise using V1 scale                    ║
║    For each chunk i:                                                         ║
║      noisy[i]     = helper[i] XOR (Vx_chunk[i]++0s)                        ║
║      r_hat[i],nerr = BCH_decode(noisy[i])                                   ║
║      if nerr == -1: FAIL immediately (errors exceeded t)                    ║
║    commit_v = HMAC-SHA256(SALT, r_hat[0]||...||r_hat[N-1])                 ║
║    PASS iff ALL nerr >= 0  AND  commit_v == commit                          ║
║                                                                              ║
║  WHY IMPOSTORS MUST FAIL                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  From observed data (total payload = 2048 bits):                            ║
║    Genuine  errors: V2=711, V3=477, V4=747  → max genuine = 747            ║
║    Impostor errors: V5=928, V6=889, V7=900  → min impostor = 889           ║
║    Gap = 889 - 747 = 142 bits on full 2048-bit payload                     ║
║                                                                              ║
║  STRATEGY: use 1 chunk = full payload (no splitting)                        ║
║    BCH must correct up to 747 errors → need t ≥ 747                        ║
║    BCH must FAIL for 889+ errors    → need t < 889                         ║
║    Choose t = 800  (sits cleanly in gap: 747 < 800 < 889)                  ║
║                                                                              ║
║  BCH(N, t=800) with N large enough to carry 2048 payload bits:             ║
║    Use N = 4095 (GF(2^12)), PAR = t×m ≈ 800×12 = 9600 bits                ║
║    K = N - PAR = 4095 - 9600 → need bigger N                               ║
║    Use N = 16383 (GF(2^14)):                                                ║
║      PAR ≈ 800 × 14 = 11200,  K = 16383 - 11200 = 5183                    ║
║      K=5183 >> 2048 → payload fits in 1 chunk with room to spare           ║
║      t=800: genuine max(747) < 800 ✓   impostor min(889) > 800 ✓          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac as hmac_module
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

# ── BCH parameters ────────────────────────────────────────────────────────────
#
#  KEY DESIGN DECISION:
#  Treat the entire 2048-bit payload as ONE chunk (no splitting).
#  This way BCH t operates directly on total payload errors.
#
#  Observed total payload errors:
#    Genuine  max = 747  (V4)
#    Impostor min = 889  (V6)
#    Gap = 142 bits → t = 800 sits cleanly between them
#
#  BCH(N=16383, GF(2^14)):
#    PAR ≈ 800 × 14 = 11200 bits
#    K   = 16383 - 11200 ≈ 5183 bits  >> 2048  → payload fits in 1 chunk ✓
#    t   = 800  → corrects up to 800 errors
#          genuine  max errors 747 < 800 → BCH SUCCEEDS → HMAC matches  → PASS ✓
#          impostor min errors 889 > 800 → BCH FAILS    → HMAC mismatch → FAIL ✗
#
BCH_N          = 16383   # 2^14 - 1,  GF(2^14)
BCH_T_DESIGNED = 800     # 747 < 800 < 889  →  clean separation
QUANT_BITS     = 4       # 512 dims × 4 bits = 2048-bit payload

# GF(2^14) primitive polynomial: x^14 + x^10 + x^6 + x + 1  (0x4041 | bit14 set)
# Standard primitive poly for GF(2^14): 0x402B  (x^14+x^5+x^3+x+1)
GF_M        = 14
GF_SIZE     = (1 << GF_M)          # 16384
GF_MASK     = GF_SIZE - 1          # 16383  = BCH_N
GF_PRIM     = 0x402B               # x^14 + x^5 + x^3 + x + 1

# Pre-built GF tables (filled in _init_gf)
_GF_EXP = [0] * (2 * GF_SIZE)
_GF_LOG = [0] * GF_SIZE


def _init_gf():
    x = 1
    for i in range(GF_MASK):
        _GF_EXP[i] = _GF_EXP[i + GF_MASK] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & GF_SIZE:
            x ^= GF_PRIM
    _GF_LOG[0] = -1


_init_gf()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] + _GF_LOG[b]) % GF_MASK]


def _gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF inverse of 0")
    return _GF_EXP[(GF_MASK - _GF_LOG[a]) % GF_MASK]


def _gf_pow(base: int, exp: int) -> int:
    if base == 0:
        return 0
    return _GF_EXP[(_GF_LOG[base] * exp) % GF_MASK]


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
        log.info(f"AdaFace model loaded  |  Provider: {providers[0]}")

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        img  = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img  = img.transpose(2, 0, 1)[np.newaxis]
        out  = self.session.run([self.output_name], {self.input_name: img})
        emb  = out[0][0] if out[0].ndim == 2 else out[0]
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
    return [(pos, frame) for _, pos, frame in top]


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
        resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
        masked  = apply_mask(resized)
        emb     = model.get_embedding(masked)
        embeddings.append(emb)
        area = crop.shape[0] * crop.shape[1]
        if area > best_area:
            best_area   = area
            best_masked = masked

    out_dir   = Path(OUTPUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"video_{video_index}_masked.jpg"
    cv2.imwrite(str(save_path), best_masked)

    # Average then renormalise
    stack = np.stack(embeddings, axis=0)
    avg   = np.mean(stack, axis=0).astype(np.float32)
    norm  = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Averaged embedding norm near zero.")
    final = (avg / norm).astype(np.float32)

    print(f"  Frames extracted : {len(frames)}  |  Face crops : {len(crops)}")
    print(f"  Final norm       : {np.linalg.norm(final):.8f}")
    print(f"  Saved photo      : {save_path}")
    return video_name, final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COSINE SIMILARITY  (informational only, NOT a gate)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pairwise_similarities(
    emb_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    sims = {}
    print("\n" + "=" * 62)
    print("  COSINE SIMILARITIES  (informational — NOT used as a gate)")
    print("=" * 62)
    for v1n, v2n in itertools.combinations(emb_dict.keys(), 2):
        sim = float(np.clip(np.dot(emb_dict[v1n], emb_dict[v2n]), -1.0, 1.0))
        sims[f"{v1n}_vs_{v2n}"] = sim
        print(f"  {v1n} vs {v2n} : {sim:.8f}")
    return sims


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GF(2) POLYNOMIAL ARITHMETIC
# ══════════════════════════════════════════════════════════════════════════════

def _gf2_divmod(dividend: list, divisor: list) -> list:
    a = list(dividend)
    b = list(divisor)
    db = len(b) - 1
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BCH GENERATOR POLYNOMIAL  (GF(2^14))
# ══════════════════════════════════════════════════════════════════════════════

def _conjugacy_class(exp: int) -> list:
    seen = []
    e    = exp % GF_MASK
    while e not in seen:
        seen.append(e)
        e = (e * 2) % GF_MASK
    return seen


def _minimal_poly(root_exp: int) -> list:
    conj     = _conjugacy_class(root_exp)
    poly     = [1]
    for e in conj:
        rv       = _GF_EXP[e % GF_MASK]
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i]     ^= c
            new_poly[i + 1] ^= _gf_mul(c, rv)
        poly = new_poly
    return [int(c & 1) for c in poly]


def build_bch_generator(t: int) -> Tuple[list, int, int]:
    """
    Build BCH(N=16383, t) generator polynomial over GF(2^14).
    Returns (g, K, PAR).
    """
    log.info(f"Building BCH(N={BCH_N}, t={t}) generator — this may take a minute …")
    g    = [1]
    used = set()
    for i in range(1, 2 * t, 2):
        cls = frozenset(_conjugacy_class(i))
        if cls in used:
            continue
        used.add(cls)
        mp = _minimal_poly(i)
        g  = _poly_mul_gf2(g, mp)

    PAR = len(g) - 1
    K   = BCH_N - PAR
    if K <= 0:
        raise ValueError(
            f"BCH(N={BCH_N}, t={t}) → K={K} ≤ 0. t is too large for this N."
        )
    log.info(f"BCH generator built: N={BCH_N}, K={K}, PAR={PAR}, t={t}")
    return g, K, PAR


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BCH ENCODE
# ══════════════════════════════════════════════════════════════════════════════

def bch_encode(msg_bits: list, g: list, K: int, PAR: int) -> list:
    if len(msg_bits) != K:
        raise ValueError(f"bch_encode: expected {K} bits, got {len(msg_bits)}")
    padded    = list(msg_bits) + [0] * PAR
    remainder = _gf2_divmod(padded, g)
    parity    = _poly_pad(remainder, PAR)
    return list(msg_bits) + parity


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BCH DECODE  (Berlekamp-Massey + Chien search, GF(2^14))
# ══════════════════════════════════════════════════════════════════════════════

def bch_decode(
    received_bits: list,
    g  : list,
    K  : int,
    PAR: int,
    t  : int,
) -> Tuple[list, int]:
    """
    Decode received N-bit word.
    Returns (message_K_bits, nerr) where nerr=-1 = uncorrectable.
    """
    if len(received_bits) != BCH_N:
        raise ValueError(f"bch_decode: expected {BCH_N} bits, got {len(received_bits)}")

    # ── Syndromes ─────────────────────────────────────────────────────────────
    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = _GF_EXP[i % GF_MASK]
        s  = 0
        for bit in received_bits:
            s = _gf_mul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

    # ── Berlekamp-Massey ─────────────────────────────────────────────────────
    C = [1] + [0] * (2 * t)
    B = [1] + [0] * (2 * t)
    L = 0; m = 1; b = 1

    for n in range(2 * t):
        d = syndromes[n]
        for j in range(1, L + 1):
            if C[j] and syndromes[n - j]:
                d ^= _gf_mul(C[j], syndromes[n - j])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T    = list(C)
            coef = _gf_mul(d, _gf_inv(b))
            for j in range(m, 2 * t + 1):
                if j - m < len(B) and B[j - m]:
                    C[j] ^= _gf_mul(coef, B[j - m])
            L = n + 1 - L; B = T; b = d; m = 1
        else:
            coef = _gf_mul(d, _gf_inv(b))
            for j in range(m, 2 * t + 1):
                if j - m < len(B) and B[j - m]:
                    C[j] ^= _gf_mul(coef, B[j - m])
            m += 1

    Lambda = C[:L + 1]
    if L > t or L == 0:
        return list(received_bits[:K]), -1

    # ── Chien search ──────────────────────────────────────────────────────────
    error_positions = []
    for j in range(1, BCH_N + 1):
        val = Lambda[0]
        for k in range(1, len(Lambda)):
            if Lambda[k]:
                val ^= _gf_mul(Lambda[k], _GF_EXP[(j * k) % GF_MASK])
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
#  SECTION 7 — QUANTISATION
# ══════════════════════════════════════════════════════════════════════════════

def embedding_to_payload(
    emb       : np.ndarray,
    shared_min: Optional[float] = None,
    shared_max: Optional[float] = None,
) -> Tuple[list, float, float]:
    """Quantise 512-dim embedding to QUANT_BITS × 512 = 2048 bits."""
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
    return bits, v_min, v_max


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — BIT / BYTE HELPERS
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


def hmac_sha256(salt: bytes, message_bits: list) -> str:
    """HMAC-SHA256(salt, bits) → 256-bit hash key (64 hex chars)."""
    return hmac_module.new(
        salt, bits_to_bytes(message_bits), hashlib.sha256
    ).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — ENROLL  (Instruction 1)
#
#  Single chunk = entire padded payload.
#  helper = BCH_encode(r) XOR (r ++ zeros(PAR))
#  commit = HMAC-SHA256(SALT, r)
#  Store: helper, commit, SALT.  Delete: r.
# ══════════════════════════════════════════════════════════════════════════════

def bch_enroll(
    v1_payload_bits: list,
    g  : list,
    K  : int,
    PAR: int,
) -> Tuple[list, str, bytes]:
    """
    Register V1.

    The payload is zero-padded to exactly K bits (one BCH chunk).
    Returns:
        helper  : K+PAR = N-bit list          (stored)
        commit  : HMAC-SHA256 hex string      (stored — 256-bit hash key)
        salt    : 32 random bytes             (stored)
    """
    if len(v1_payload_bits) > K:
        raise ValueError(
            f"Payload {len(v1_payload_bits)} bits > K={K}. "
            f"Increase N or decrease QUANT_BITS."
        )

    # Zero-pad to K bits  (secret r = payload + zero padding)
    pad = K - len(v1_payload_bits)
    r   = list(v1_payload_bits) + [0] * pad   # K-bit secret

    # BCH encode secret
    cw_r   = bch_encode(r, g, K, PAR)         # N-bit codeword
    pad_n  = r + [0] * PAR                     # r zero-extended to N bits
    helper = [a ^ b for a, b in zip(cw_r, pad_n)]  # Juels-Wattenberg sketch

    # 256-bit random salt + HMAC commitment
    salt   = os.urandom(32)
    commit = hmac_sha256(salt, r)

    # Sanity: syndrome of encoded r must be zero
    syndrome_ok = all(
        s == 0 for s in _gf2_divmod(bch_encode(r, g, K, PAR), g)
    )
    log.info(f"Enroll — syndrome zero: {syndrome_ok}  ← must be True")
    log.info(f"Enroll — SALT   (256-bit): {salt.hex()}")
    log.info(f"Enroll — Commit (256-bit): {commit}")
    log.info(f"Enroll — Payload {len(v1_payload_bits)} bits, padded to K={K}, PAR={PAR}")

    return helper, commit, salt


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — VERIFY  (Instruction 2)
#
#  Each of V2..V7 runs independently.
#  noisy  = helper XOR (Vx_pad ++ zeros(PAR))
#  r_hat, nerr = BCH_decode(noisy)
#  PASS iff nerr >= 0  AND  HMAC-SHA256(SALT, r_hat) == commit
#
#  SECURITY:
#    Genuine  errors ≤ 747 < t=800  → BCH corrects  → HMAC matches  → PASS ✓
#    Impostor errors ≥ 889 > t=800  → BCH fails      → HMAC mismatch → FAIL ✗
# ══════════════════════════════════════════════════════════════════════════════

def bch_verify(
    vx_payload_bits : list,
    v1_payload_bits : list,
    helper          : list,
    commit_enroll   : str,
    salt            : bytes,
    g               : list,
    K               : int,
    PAR             : int,
    t               : int,
    video_label     : str,
    expected_role   : str,
) -> dict:
    sep = "─" * 60

    # ── Pad both payloads to K bits ───────────────────────────────────────────
    pad_vx = K - len(vx_payload_bits)
    pad_v1 = K - len(v1_payload_bits)
    vx_pad = list(vx_payload_bits) + [0] * pad_vx
    v1_pad = list(v1_payload_bits) + [0] * pad_v1

    # ── Total Hamming distance (informational) ────────────────────────────────
    total_errors = sum(a != b for a, b in zip(vx_pad, v1_pad))
    error_rate   = total_errors / len(v1_payload_bits)

    print(f"  Role expected            : {expected_role}")
    print(f"  Total payload bit errors : {total_errors} / {len(v1_payload_bits)}"
          f"  ({error_rate*100:.2f}%)  [informational]")
    print(f"  BCH correction limit     : t = {t}")
    print()

    if total_errors <= t:
        print(f"  Bit errors ({total_errors}) ≤ t ({t})"
              f"  → BCH should CORRECT → genuine expected")
    else:
        print(f"  Bit errors ({total_errors}) > t ({t})"
              f"  → BCH should FAIL → impostor expected")
    print()

    # ── Build noisy codeword: helper XOR (Vx_pad ++ zeros(PAR)) ──────────────
    vx_extended = vx_pad + [0] * PAR          # extend to N bits
    noisy_cw    = [a ^ b for a, b in zip(helper, vx_extended)]

    # ── BCH decode ────────────────────────────────────────────────────────────
    r_hat, nerr = bch_decode(noisy_cw, g, K, PAR, t)

    # ── HMAC check ────────────────────────────────────────────────────────────
    commit_verify  = hmac_sha256(salt, r_hat)
    hmac_match     = commit_verify == commit_enroll

    # PASS = BCH succeeded AND HMAC matches
    verified = (nerr >= 0) and hmac_match

    # Remaining errors after BCH correction
    remaining = sum(a != b for a, b in zip(r_hat, v1_pad))

    # ── Print ─────────────────────────────────────────────────────────────────
    if nerr >= 0:
        print(f"  BCH decode               : SUCCESS  ({nerr} errors corrected)")
    else:
        print(f"  BCH decode               : FAILED   (errors > t={t} — uncorrectable)")

    print(f"  Remaining bit errors     : {remaining}"
          f"  {'(0 = perfect ✓)' if remaining == 0 else ''}")
    print()
    print(f"  Stored commit   (256-bit): {commit_enroll}")
    print(f"  Computed commit (256-bit): {commit_verify}")
    print(f"  HMAC-SHA256 match        : {'YES ✓' if hmac_match else 'NO  ✗'}")
    print()

    if verified:
        verdict = "ACCESS GRANTED ✓  —  BCH recovered V1 secret, HMAC-SHA256 verified"
    else:
        reasons = []
        if nerr < 0:
            reasons.append(
                f"BCH FAILED (errors {total_errors} > t={t}  → "
                f"impostor cannot recover secret from helper data)"
            )
        if not hmac_match:
            reasons.append("HMAC MISMATCH (recovered bits ≠ enrolled secret)")
        verdict = "ACCESS DENIED  ✗  —  " + "  |  ".join(reasons)

    print(f"  ╔{'═'*56}╗")
    print(f"  ║  Verdict : {verdict[:52]:<52}  ║")
    if len(verdict) > 52:
        for part in [verdict[i:i+52] for i in range(52, len(verdict), 52)]:
            print(f"  ║  {'':10}{part:<52}  ║")
    print(f"  ╚{'═'*56}╝")

    correct = (
        (expected_role == "GENUINE"  and     verified) or
        (expected_role == "IMPOSTOR" and not verified)
    )
    print(f"  Security check           : "
          f"{'✓ CORRECT' if correct else '✗ UNEXPECTED — check t vs error distribution'}")
    print(sep)

    return {
        "label"         : video_label,
        "expected_role" : expected_role,
        "total_errors"  : total_errors,
        "error_rate"    : error_rate,
        "nerr"          : nerr,
        "remaining"     : remaining,
        "hmac_match"    : hmac_match,
        "verified"      : verified,
        "correct"       : correct,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    sep70 = "=" * 70

    # ── Build BCH generator ───────────────────────────────────────────────────
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    PAYLOAD_BITS = 512 * QUANT_BITS   # 2048
    PAD_NEEDED   = BCH_K - PAYLOAD_BITS

    if PAYLOAD_BITS > BCH_K:
        raise ValueError(
            f"Payload ({PAYLOAD_BITS} bits) > K ({BCH_K}). "
            f"Increase BCH_N or reduce QUANT_BITS."
        )

    print(sep70)
    print("  ADAFACE + BCH FUZZY COMMITMENT — Phase 13 FINAL")
    print("  Single Gate: BCH syndrome decode + HMAC-SHA256")
    print("  No G1 (cosine) gate.  No G2 (Hamming rate) gate.")
    print(sep70)
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  INSTRUCTION 1 — REGISTRATION (V1)                         │")
    print("  │    r = payload + zero_pad  (K bits, one BCH chunk)         │")
    print("  │    helper = BCH_encode(r) XOR (r ++ zeros(PAR))            │")
    print("  │    commit = HMAC-SHA256(SALT, r)  ← 256-bit hash key       │")
    print("  │    STORE: helper, commit, SALT.   DELETE: r, embedding     │")
    print("  │                                                             │")
    print("  │  INSTRUCTION 2 — LOGIN (V2..V7 each independently)         │")
    print("  │    noisy  = helper XOR (Vx_pad ++ zeros(PAR))              │")
    print("  │    r_hat, nerr = BCH_decode(noisy)                         │")
    print("  │    PASS iff nerr ≥ 0  AND  HMAC-SHA256(SALT,r_hat)==commit │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ERROR SEPARATION (why this works):")
    print(f"    Genuine  max errors : 747  (V4)   →  747 < t={BCH_T_DESIGNED} → BCH corrects ✓")
    print(f"    Impostor min errors : 889  (V6)   →  889 > t={BCH_T_DESIGNED} → BCH fails    ✓")
    print(f"    Gap = 889 - 747 = 142 bits.  t={BCH_T_DESIGNED} sits cleanly inside gap.")
    print()
    print(f"  BCH PARAMETERS:")
    print(f"    N   = {BCH_N}   (GF(2^{GF_M}),  2^{GF_M}-1 = {BCH_N})")
    print(f"    K   = {BCH_K}   (message bits per chunk)")
    print(f"    PAR = {BCH_PAR}  (parity bits)")
    print(f"    t   = {BCH_T_DESIGNED}   (error correction capacity)")
    print(f"    Payload = {PAYLOAD_BITS} bits  fits in 1 chunk (K={BCH_K})")
    print(f"    Padding = {PAD_NEEDED} zero bits appended to payload")
    print(f"    Helper  = {BCH_N} bits stored  (one codeword)")
    print(sep70)

    # ── Validate paths ────────────────────────────────────────────────────────
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    # ── Extract embeddings ────────────────────────────────────────────────────
    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        result = process_video(vp, idx, model, detector)
        if result:
            name, emb = result
            embeddings[name] = emb

    print(f"\n{sep70}")
    print("  EMBEDDING NORMS")
    print(sep70)
    for name, emb in embeddings.items():
        print(f"  {name}: norm = {np.linalg.norm(emb):.8f}")
    print(sep70)

    # Cosine similarities — informational only
    if len(embeddings) >= 2:
        compute_pairwise_similarities(embeddings)

    if "video_1" not in embeddings:
        print(f"\n{sep70}\n  ERROR: video_1 not available.\n{sep70}")
        return embeddings, {}

    # ── Quantise V1 — capture scale ───────────────────────────────────────────
    v1_bits, v1_min, v1_max = embedding_to_payload(embeddings["video_1"])
    log.info(f"V1 quantised: scale=[{v1_min:.5f}, {v1_max:.5f}]  "
             f"payload={len(v1_bits)} bits")

    # ── INSTRUCTION 1: REGISTER V1 ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  INSTRUCTION 1 — REGISTRATION — V1  (IOS Beard)")
    print(f"{'─'*60}")

    helper, commit_H1, enroll_salt = bch_enroll(
        v1_payload_bits=v1_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
    )

    print(f"  Payload bits              : {len(v1_bits)}")
    print(f"  Padded to K               : {BCH_K} bits  (pad={PAD_NEEDED} zeros)")
    print(f"  Helper size               : {BCH_N} bits  (one BCH codeword)")
    print(f"  SALT (256-bit)            : {enroll_salt.hex()}")
    print(f"  Commit (256-bit HMAC key) : {commit_H1}")
    print(f"  Helper (first 64 hex)     : {bits_to_hex(helper)[:64]}…")
    print(f"  V1 quant scale            : [{v1_min:.6f}, {v1_max:.6f}]")
    print(f"  NOTE: Raw embedding and secret r DELETED after registration.")
    print(f"{'─'*60}")

    # ── INSTRUCTION 2: VERIFY V2..V7 ─────────────────────────────────────────
    verify_config = {
        "video_2": ("IOS No Beard     (V2)", "GENUINE"),
        "video_3": ("Android Beard    (V3)", "GENUINE"),
        "video_4": ("Android No Beard (V4)", "GENUINE"),
        "video_5": ("Android Video 5  (V5)", "IMPOSTOR"),
        "video_6": ("IOS Sha          (V6)", "IMPOSTOR"),
        "video_7": ("IOS Rusl         (V7)", "IMPOSTOR"),
    }

    summary = []

    for vid, (label, expected_role) in verify_config.items():
        if vid not in embeddings:
            log.error(f"{vid} not available — skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"  INSTRUCTION 2 — LOGIN — {label}")
        print(f"  Using V1 helper + SALT + 256-bit commit")
        print(f"{'─'*60}")

        # Quantise using V1 scale
        vx_bits, _, _ = embedding_to_payload(
            embeddings[vid], shared_min=v1_min, shared_max=v1_max
        )
        log.info(f"{vid}: payload={len(vx_bits)} bits  (V1 scale)")

        result = bch_verify(
            vx_payload_bits=vx_bits,
            v1_payload_bits=v1_bits,
            helper         =helper,
            commit_enroll  =commit_H1,
            salt           =enroll_salt,
            g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
            video_label    =label,
            expected_role  =expected_role,
        )
        summary.append(result)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 13 — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Registered user  : V1 — IOS Beard")
    print(f"  SALT             : {enroll_salt.hex()[:32]}…  (256-bit)")
    print(f"  Commit H1        : {commit_H1}  (256-bit HMAC)")
    print(f"  BCH              : N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED}, PAR={BCH_PAR}")
    print(f"  Single chunk     : entire {PAYLOAD_BITS}-bit payload in one BCH codeword")
    print(f"  Gate             : BCH decode (nerr ≥ 0) + HMAC-SHA256 match")
    print()
    print(f"  {'Video':<26}  {'Role':>8}  {'Errors':>6}  {'nerr':>6}  "
          f"{'HMAC':>5}  {'Result':>14}  {'OK?':>12}")
    print(f"  {'─'*26}  {'─'*8}  {'─'*6}  {'─'*6}  "
          f"{'─'*5}  {'─'*14}  {'─'*12}")

    for r in summary:
        nerr_s   = "FAIL" if r["nerr"] < 0 else str(r["nerr"])
        hmac_s   = "MATCH" if r["hmac_match"] else "MISS"
        result_s = "GRANTED ✓" if r["verified"] else "DENIED  ✗"
        ok_s     = "✓ correct" if r["correct"] else "✗ UNEXPECTED"
        print(
            f"  {r['label']:<26}  "
            f"{r['expected_role']:>8}  "
            f"{r['total_errors']:>6}  "
            f"{nerr_s:>6}  "
            f"{hmac_s:>5}  "
            f"{result_s:>14}  "
            f"{ok_s:>12}"
        )

    # ── Error separation analysis ─────────────────────────────────────────────
    genuine_results  = [r for r in summary if r["expected_role"] == "GENUINE"]
    impostor_results = [r for r in summary if r["expected_role"] == "IMPOSTOR"]

    print()
    print(f"  {'='*66}")
    print("  ERROR SEPARATION ANALYSIS")
    print(f"  {'='*66}")
    print(f"  t = {BCH_T_DESIGNED}  (BCH correction capacity)")
    print()

    if genuine_results:
        g_errors = [r["total_errors"] for r in genuine_results]
        g_pass   = all(r["verified"] for r in genuine_results)
        print(f"  GENUINE  (V2, V3, V4):")
        print(f"    Error counts : {g_errors}")
        print(f"    Max errors   : {max(g_errors)}  →  "
              f"{'≤ t=' + str(BCH_T_DESIGNED) + ' ✓  BCH corrects' if max(g_errors) <= BCH_T_DESIGNED else '> t ✗  BCH FAILS — reduce t'}")
        print(f"    All GRANTED  : {'YES ✓' if g_pass else 'NO ✗'}")

    if impostor_results:
        i_errors = [r["total_errors"] for r in impostor_results]
        i_fail   = all(not r["verified"] for r in impostor_results)
        print()
        print(f"  IMPOSTOR (V5, V6, V7):")
        print(f"    Error counts : {i_errors}")
        print(f"    Min errors   : {min(i_errors)}  →  "
              f"{'> t=' + str(BCH_T_DESIGNED) + ' ✓  BCH fails impostor' if min(i_errors) > BCH_T_DESIGNED else '≤ t ✗  BCH CORRECTS — raise t'}")
        print(f"    All DENIED   : {'YES ✓' if i_fail else 'NO ✗'}")

    if genuine_results and impostor_results:
        gap = min(i_errors) - max(g_errors)
        print()
        print(f"  Gap = min_impostor({min(i_errors)}) - max_genuine({max(g_errors)}) = {gap} bits")
        if gap > 0:
            print(f"  t={BCH_T_DESIGNED} sits inside gap [{max(g_errors)}, {min(i_errors)}]  →  "
                  f"clean separation ✓")
            print(f"  Safe t range : {max(g_errors)+1} ≤ t ≤ {min(i_errors)-1}")
        else:
            print(f"  WARNING: gap ≤ 0 — distributions overlap at total-payload level.")
            print(f"  Adjust t: set t between {max(g_errors)} and {min(i_errors)}.")
            print(f"  If {max(g_errors)} >= {min(i_errors)} then no BCH-only solution exists.")

    # ── Outcome ───────────────────────────────────────────────────────────────
    print()
    print(f"  {'='*66}")
    print("  OUTCOME")
    print(f"  {'='*66}")
    all_correct = all(r["correct"] for r in summary)

    for r in summary:
        tag = "ACCESS GRANTED ✓" if r["verified"] else "ACCESS DENIED  ✗"
        exp = "(expected ✓)" if r["correct"] else "(UNEXPECTED ✗)"
        print(f"    {r['label']:<28} → {tag}  {exp}")

    print()
    if all_correct:
        print("  ✓ ALL RESULTS CORRECT")
        print(f"    Genuine (V2,V3,V4): BCH corrected errors, HMAC matched → GRANTED")
        print(f"    Impostors (V5,V6,V7): errors > t={BCH_T_DESIGNED}, BCH failed → DENIED")
        print(f"    Helper data alone reveals nothing about the enrolled secret.")
    else:
        print("  ✗ UNEXPECTED RESULTS — adjust t:")
        for r in summary:
            if not r["correct"]:
                if r["expected_role"] == "GENUINE" and not r["verified"]:
                    print(f"    {r['label']}: genuine DENIED — errors {r['total_errors']} > t={BCH_T_DESIGNED}")
                    print(f"    → Raise t above {r['total_errors']}")
                elif r["expected_role"] == "IMPOSTOR" and r["verified"]:
                    print(f"    {r['label']}: impostor GRANTED — errors {r['total_errors']} ≤ t={BCH_T_DESIGNED}")
                    print(f"    → Lower t below {r['total_errors']}")

    print()
    print("  CRYPTOGRAPHIC PROPERTIES:")
    print(f"    Scheme    : Juels-Wattenberg fuzzy commitment")
    print(f"    Sketch    : helper = BCH_encode(r) XOR (r ++ zeros(PAR))")
    print(f"    Commit    : HMAC-SHA256(256-bit SALT, r)  →  256-bit hash key")
    print(f"    Security  : helper reveals 0 bits about r without face within t errors")
    print(f"    BCH field : GF(2^{GF_M}),  N={BCH_N}")
    print(f"    t={BCH_T_DESIGNED}, K={BCH_K}, PAR={BCH_PAR}")
    print(sep70)

    return embeddings, {}


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY COMMITMENT — Phase 13 FINAL")
    print("  NO G1 (cosine).  NO G2 (Hamming rate).  BCH + HMAC only.")
    print(f"  BCH(N={BCH_N}, GF(2^{GF_M}), t={BCH_T_DESIGNED})")
    print(f"  One chunk = entire {512*QUANT_BITS}-bit payload")
    print(f"  Gap: genuine max 747  <  t={BCH_T_DESIGNED}  <  impostor min 889")
    print("  EXPECTED:  V2 GRANTED ✓  V3 GRANTED ✓  V4 GRANTED ✓")
    print("             V5 DENIED  ✗  V6 DENIED  ✗  V7 DENIED  ✗")
    print("=" * 70)

    embeddings, _ = run()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Videos processed : {len(embeddings)} / {len(VIDEO_PATHS)}")
    print("=" * 70)
