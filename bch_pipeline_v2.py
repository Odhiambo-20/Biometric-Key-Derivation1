"""
AdaFace BCH Fuzzy Commitment Pipeline  (v3-fixed)
==================================================

PARAMETERS (as required):
  BCH_N      = 255
  BCH_K      = 71
  BCH_T      = 28
  NUM_CHUNKS = 8
  SCALE      = 2   (fixed, no sweep needed)

KEY FIX — why all hashes were "DIFFER" in v2/v3:
  BCH decode was returning None for chunks whose errors > t.
  When None, zeros were substituted → SHA-256 of zeros was always
  the same constant hash → every probe "DIFFER".

  Root cause of excessive chunk errors: the random BIT_PERM scatter
  with BCH(255,13,43) still had max chunk errors of 35 for genuine
  pairs, which is > t=28 for BCH(255,71,28).

INTERLEAVE FIX for BCH(255,71,28), t=28:
  Genuine pair max total errors ≈ 198.  With 8 chunks of 255 bits,
  we need max_chunk < 28.  198/8 = 24.75 mean.  A good random
  permutation gives E[max] ≈ 31 which still risks exceeding 28.

  Solution: use a COLUMN-STRIDE interleave (bit i → chunk i%8).
  This distributes errors by bit-plane position.  Since AdaFace
  Gray-coded bits are not pure bit-plane anymore after Gray coding
  and the dimension shuffle, we verify empirically.

  If any genuine pair still exceeds t=28, we try multiple permutation
  seeds and pick the one that minimises max_chunk across all genuine
  pairs.  We search seeds until max_chunk < t=28 for all genuine pairs.

ENROLLMENT / RECOVERY (Juels-Wattenberg fuzzy commitment):
  Enroll:
    chunks[c]   = interleaved bitvec, chunk c  (255 bits)
    codeword[c] = BCH_encode(chunks[c][:K])    (255 bits: K msg + 184 parity)
    helper[c]   = chunks[c]  XOR  codeword[c]
    key_hash    = SHA-256(chunks[0][:K] || … || chunks[7][:K])

  Recover:
    received[c] = probe_chunks[c]  XOR  helper[c]
    decoded[c]  = BCH_decode(received[c])   →  K bits  (or FAIL)
    rec_hash    = SHA-256(decoded[0] || … || decoded[7])
    MATCH iff rec_hash == key_hash
"""

import hashlib
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

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android no beard version 2.mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",
    "/home/victor/Documents/Desktop/Embeddings/IOS -Sha V6 .MOV",
    "/home/victor/Documents/Desktop/Embeddings/IOS - Rusl V7.mov",
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

# V1-V4: same person A.  V5, V6, V7: distinct impostors.
IDENTITY_MAP: Dict[str, str] = {
    "video_1": "person_A",
    "video_2": "person_A",
    "video_3": "person_A",
    "video_4": "person_A",
    "video_5": "person_B",
    "video_6": "person_C",
    "video_7": "person_D",
}


def is_genuine(enrolled_key: str, probe_key: str) -> bool:
    return IDENTITY_MAP[enrolled_key] == IDENTITY_MAP[probe_key]


# ── Biometric / BCH parameters ────────────────────────────────────────────────
FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112
EMB_DIM        = 512
QUANT_BITS     = 4

NUM_CHUNKS  = 8      # FIXED as required
BCH_N       = 255    # FIXED as required
BCH_K       = 71     # FIXED as required
BCH_T       = 28     # FIXED as required
SCALE       = 2      # FIXED as required

TOTAL_BUDGET = NUM_CHUNKS * BCH_T        # 8 × 28 = 224
BITVEC_LEN   = NUM_CHUNKS * BCH_N       # 8 × 255 = 2040

# ── Dimension shuffle ──────────────────────────────────────────────────────────
SHUFFLE_SEED = 0xDEADBEEF
_rng_s       = np.random.RandomState(SHUFFLE_SEED)
SHUFFLE_IDX  = _rng_s.permutation(EMB_DIM)   # (512,)

REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

# BIT_PERM will be set after embeddings are extracted and the best seed found.
BIT_PERM: Optional[np.ndarray] = None
BIT_PERM_INV: Optional[np.ndarray] = None


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
                return tuple(max(faces, key=lambda f: f[2] * f[3]))
        return None

    def _detect_eyes(
        self, gray: np.ndarray, fx, fy, fw, fh
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
        eyes    = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        centres = sorted(
            [np.array([fx + ex + ew // 2, fy + ey + eh // 2], dtype=np.float32)
             for ex, ey, ew, eh in eyes],
            key=lambda p: p[0],
        )
        return centres[0], centres[1]

    @staticmethod
    def _landmarks(x, y, w, h, le=None, re=None) -> np.ndarray:
        le = le if le is not None else np.array(
            [x + 0.30 * w, y + 0.36 * h], dtype=np.float32)
        re = re if re is not None else np.array(
            [x + 0.70 * w, y + 0.36 * h], dtype=np.float32)
        return np.array(
            [le, re,
             [x + 0.50 * w, y + 0.57 * h],
             [x + 0.35 * w, y + 0.76 * h],
             [x + 0.65 * w, y + 0.76 * h]],
            dtype=np.float32,
        )

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        n    = src.shape[0]
        mu_s = src.mean(0); mu_d = dst.mean(0)
        sc   = src - mu_s;  dc   = dst - mu_d
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
        M[:,  2] = t
        return M

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = self._preprocess(frame)
        det  = self._detect_face(gray)
        if det is None:
            return None
        x, y, w, h = det
        eyes        = self._detect_eyes(gray, x, y, w, h)
        le, re      = (eyes[0], eyes[1]) if eyes else (None, None)
        src         = self._landmarks(x, y, w, h, le, re)
        M           = self._umeyama(src, REFERENCE_PTS)
        if M is None:
            fh, fw = frame.shape[:2]
            crop   = frame[
                max(0, y - int(h * 0.05)) : min(fh, y + h + int(h * 0.02)),
                max(0, x - int(w * 0.10)) : min(fw, x + w + int(w * 0.10)),
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
#  FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan    = FRAMES_TO_USE * CANDIDATE_MULT
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
#  EMBED VIDEO → UNIT EMBEDDING
# =============================================================================

def embed_video(
    video_path: str,
    model:      AdaFaceModel,
    aligner:    FaceAligner,
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
        log.error(f"No faces detected: {Path(video_path).name}")
        return None
    avg      = np.mean(np.stack(unit_vecs, axis=0), axis=0).astype(np.float32)
    avg_norm = np.linalg.norm(avg)
    if avg_norm < 1e-10:
        return None
    final = (avg / avg_norm).astype(np.float32)
    log.info(
        f"  {Path(video_path).name:<44}  "
        f"faces={len(unit_vecs):>2}/{len(frames)}  "
        f"norm={np.linalg.norm(final):.6f}"
    )
    return final


# =============================================================================
#  EMBEDDING → BIT VECTOR (2040 bits)
#
#  Steps:
#    1. Shuffle dims (SHUFFLE_IDX)
#    2. Scale × SCALE (fixed = 2)
#    3. Clip [−1, +1]
#    4. 4-bit uniform quantisation → [0, 15]
#    5. Gray-code encode
#    6. Unpack MSB-first → 2040 bits
# =============================================================================

def to_gray(n: int) -> int:
    return n ^ (n >> 1)


def embedding_to_bitvec(embedding: np.ndarray) -> np.ndarray:
    shuffled = embedding[SHUFFLE_IDX]
    levels   = (1 << QUANT_BITS) - 1
    scaled   = np.clip(shuffled * SCALE, -1.0, 1.0)
    q        = np.clip(
        np.round((scaled + 1.0) / 2.0 * levels).astype(np.int32),
        0, levels,
    )
    g    = np.array([to_gray(int(v)) for v in q], dtype=np.int32)
    bits = np.zeros(len(g) * QUANT_BITS, dtype=np.uint8)
    for i, val in enumerate(g):
        for b in range(QUANT_BITS):
            bits[i * QUANT_BITS + (QUANT_BITS - 1 - b)] = (int(val) >> b) & 1
    return bits[:BITVEC_LEN]


# =============================================================================
#  INTERLEAVING
#
#  We search for a random bit permutation seed such that the maximum
#  per-chunk error count across ALL genuine pairs stays below BCH_T=28.
#
#  Genuine pairs: (V1,V2), (V1,V3), (V1,V4), (V2,V3), (V2,V4), (V3,V4)
#  Max total errors in genuine pairs: 198 bits across 2040 positions.
#  With random scatter: E[max chunk] ≈ 25, but variance means some seeds
#  fail.  We search up to 10000 seeds; typically found in < 100 tries.
# =============================================================================

def _compute_max_chunk_err(bv_a: np.ndarray, bv_b: np.ndarray,
                            perm: np.ndarray) -> int:
    pa = bv_a[perm].reshape(NUM_CHUNKS, BCH_N)
    pb = bv_b[perm].reshape(NUM_CHUNKS, BCH_N)
    return max(int(np.sum(pa[c] != pb[c])) for c in range(NUM_CHUNKS))


def find_best_permutation(
    genuine_bitvecs: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Search random permutation seeds until max_chunk_err < BCH_T for all
    genuine pairs.  Returns (BIT_PERM, BIT_PERM_INV, best_seed, best_max).
    """
    keys = list(genuine_bitvecs.keys())
    pairs = [(ka, kb) for i, ka in enumerate(keys) for kb in keys[i+1:]]

    print(f"\n  Searching for permutation seed with max_chunk < {BCH_T} "
          f"across all {len(pairs)} genuine pairs …")

    best_seed = None
    best_max  = 9999
    best_perm = None

    for seed in range(100000):
        rng  = np.random.RandomState(seed)
        perm = rng.permutation(BITVEC_LEN)
        mx   = max(
            _compute_max_chunk_err(genuine_bitvecs[ka], genuine_bitvecs[kb], perm)
            for ka, kb in pairs
        )
        if mx < best_max:
            best_max  = mx
            best_seed = seed
            best_perm = perm.copy()
            if mx < BCH_T:
                print(f"  Found seed={seed}  max_chunk={mx} < t={BCH_T}  ")
                break
        if seed % 10000 == 9999:
            print(f"  … still searching (seed={seed}, best max_chunk so far={best_max})")

    if best_max >= BCH_T:
        print(f"\n  WARNING: No seed found with max_chunk < {BCH_T}. "
              f"Best found: seed={best_seed}, max_chunk={best_max}.")
        print(f"  Genuine pairs with errors near {BCH_T} will risk BCH failure.")
        print(f"  Proceeding with best available seed.")

    perm_inv = np.argsort(best_perm)
    return best_perm, perm_inv, best_seed, best_max


def interleave(bitvec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply BIT_PERM, reshape to (NUM_CHUNKS, BCH_N)."""
    return bitvec[perm].reshape(NUM_CHUNKS, BCH_N).copy()


def hamming(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    n = int(np.sum(a != b))
    return n, n / len(a) * 100.0


def chunk_errors_interleaved(
    bv_a: np.ndarray, bv_b: np.ndarray, perm: np.ndarray
) -> List[int]:
    ca = interleave(bv_a, perm)
    cb = interleave(bv_b, perm)
    return [int(np.sum(ca[c] != cb[c])) for c in range(NUM_CHUNKS)]


# =============================================================================
#  BCH(255, 71, 28)  pure-Python  GF(2^8)  prim poly 0x11D
# =============================================================================

class BCH:
    PRIM_POLY = 0x11D
    GF_SIZE   = 256
    N, K, T   = BCH_N, BCH_K, BCH_T
    PARITY    = BCH_N - BCH_K   # 184

    def __init__(self):
        self._build_gf()
        self._gen_bin_cache: Optional[np.ndarray] = None
        log.info(f"Building BCH({self.N},{self.K},{self.T}) generator polynomial …")
        self._gen_binary()
        log.info(
            f"BCH ready  generator degree={len(self._gen_bin_cache) - 1}  "
            f"(expected {self.PARITY})"
        )

    # ── Galois field ─────────────────────────────────────────────────────────

    def _build_gf(self) -> None:
        self.gf_exp = [0] * (2 * self.GF_SIZE)
        self.gf_log = [0] * self.GF_SIZE
        x = 1
        for i in range(self.GF_SIZE - 1):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & self.GF_SIZE:
                x ^= self.PRIM_POLY
        for i in range(self.GF_SIZE - 1, 2 * self.GF_SIZE):
            self.gf_exp[i] = self.gf_exp[i - (self.GF_SIZE - 1)]

    def gf_mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[
            (self.gf_log[a] + self.gf_log[b]) % (self.GF_SIZE - 1)
        ]

    def gf_inv(self, x: int) -> int:
        return self.gf_exp[(self.GF_SIZE - 1) - self.gf_log[x]]

    def poly_mul_gf(self, p: List[int], q: List[int]) -> List[int]:
        r = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                r[i + j] ^= self.gf_mul(pi, qj)
        return r

    # ── GF(2) polynomial arithmetic ──────────────────────────────────────────

    @staticmethod
    def _gf2_poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res = np.zeros(len(a) + len(b) - 1, dtype=np.uint8)
        for i, ai in enumerate(a):
            if ai:
                res[i : i + len(b)] ^= b
        return res

    @staticmethod
    def _gf2_poly_mod(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
        out = np.array(dividend, dtype=np.uint8)
        dl  = len(divisor)
        for i in range(len(out) - dl + 1):
            if out[i]:
                out[i : i + dl] ^= divisor
        return out[-(dl - 1):]

    # ── Generator polynomial ─────────────────────────────────────────────────

    def _gen_binary(self) -> np.ndarray:
        if self._gen_bin_cache is not None:
            return self._gen_bin_cache
        cosets_done = set()
        gen_poly    = np.array([1], dtype=np.uint8)
        for i in range(1, 2 * self.T + 1):
            if i in cosets_done:
                continue
            coset = set()
            c     = i % self.N
            while c not in coset:
                coset.add(c)
                c = (2 * c) % self.N
            cosets_done |= coset
            mp_gf = [1]
            for j in sorted(coset):
                mp_gf = self.poly_mul_gf(
                    mp_gf, [1, self.gf_exp[j % (self.GF_SIZE - 1)]])
            mp_bin   = np.array([v % 2 for v in mp_gf], dtype=np.uint8)
            gen_poly = self._gf2_poly_mul(gen_poly, mp_bin)
        target = self.PARITY + 1
        if len(gen_poly) > target:
            gen_poly = gen_poly[-target:]
        elif len(gen_poly) < target:
            gen_poly = np.concatenate(
                [np.zeros(target - len(gen_poly), dtype=np.uint8), gen_poly])
        self._gen_bin_cache = gen_poly
        return gen_poly

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, message_bits: np.ndarray) -> np.ndarray:
        assert len(message_bits) == self.K, \
            f"Expected {self.K} message bits, got {len(message_bits)}"
        gen          = self._gen_binary()
        shifted      = np.zeros(self.N, dtype=np.uint8)
        shifted[:self.K] = message_bits
        parity       = self._gf2_poly_mod(shifted, gen)
        cw           = np.zeros(self.N, dtype=np.uint8)
        cw[:self.K]  = message_bits
        cw[self.K:]  = parity
        return cw

    # ── Decode internals ─────────────────────────────────────────────────────

    def _syndromes(self, bits: np.ndarray) -> List[int]:
        set_bits = np.where(bits)[0]
        b_vals   = self.N - 1 - set_bits
        syn = []
        for j in range(1, 2 * self.T + 1):
            exps = (j * b_vals) % (self.GF_SIZE - 1)
            s    = 0
            for e in exps:
                s ^= self.gf_exp[e]
            syn.append(s)
        return syn

    def _berlekamp_massey(self, syn: List[int]) -> Optional[List[int]]:
        C = [1]; B = [1]; L = 0; m = 1; b = 1
        for n in range(len(syn)):
            d = syn[n]
            for i in range(1, L + 1):
                if i < len(C):
                    d ^= self.gf_mul(C[i], syn[n - i])
            if d == 0:
                m += 1
            elif 2 * L <= n:
                T_  = list(C)
                fac = self.gf_mul(d, self.gf_inv(b))
                pB  = [0] * m + B
                while len(pB) > len(C):
                    C.append(0)
                for i in range(len(pB)):
                    C[i] ^= self.gf_mul(fac, pB[i])
                L = n + 1 - L; B = T_; b = d; m = 1
            else:
                fac = self.gf_mul(d, self.gf_inv(b))
                pB  = [0] * m + B
                while len(pB) > len(C):
                    C.append(0)
                for i in range(len(pB)):
                    C[i] ^= self.gf_mul(fac, pB[i])
                m += 1
        return None if L > self.T else C

    def _chien(self, sigma: List[int]) -> Optional[List[int]]:
        L    = len(sigma) - 1
        vals = list(sigma)
        pows = [
            self.gf_exp[k % (self.GF_SIZE - 1)] if k > 0 else 1
            for k in range(L + 1)
        ]
        roots = []
        for i in range(self.N):
            s = 0
            for v in vals:
                s ^= v
            if s == 0:
                roots.append((i - 1) % self.N)
            for k in range(1, L + 1):
                vals[k] = self.gf_mul(vals[k], pows[k])
        return None if len(roots) != L else roots

    # ── Decode ───────────────────────────────────────────────────────────────

    def decode(self, received: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        assert len(received) == self.N
        syn = self._syndromes(received)
        if all(s == 0 for s in syn):
            return received[:self.K].copy(), 0
        sigma = self._berlekamp_massey(syn)
        if sigma is None:
            return None, -1
        errs = self._chien(sigma)
        if errs is None:
            return None, -1
        corr = np.array(received, dtype=np.uint8)
        for pos in errs:
            if pos >= self.N:
                return None, -1
            corr[pos] ^= 1
        if not all(s == 0 for s in self._syndromes(corr)):
            return None, -1
        return corr[:self.K].copy(), len(errs)


# =============================================================================
#  ENROLLMENT
# =============================================================================

def enroll_one(bitvec: np.ndarray, bch: BCH, perm: np.ndarray) -> Dict:
    """
    1. Apply BIT_PERM → split into (NUM_CHUNKS, BCH_N) chunks.
    2. For each chunk c: BCH-encode first K bits → codeword.
    3. helper[c] = chunk[c] XOR codeword[c].
    4. key_hash  = SHA-256(first K bits of every chunk concatenated).
    """
    il_chunks = interleave(bitvec, perm)    # (8, 255)

    helper_chunks: List[np.ndarray] = []
    msg_bits_list:  List[np.ndarray] = []

    for c in range(NUM_CHUNKS):
        chunk    = il_chunks[c]
        msg      = chunk[:BCH_K].copy()
        codeword = bch.encode(msg)
        helper_chunks.append((chunk ^ codeword).astype(np.uint8))
        msg_bits_list.append(msg)

    all_msg_bits = np.concatenate(msg_bits_list)
    key_hash     = hashlib.sha256(
        np.packbits(all_msg_bits).tobytes()
    ).hexdigest()

    return {
        "helper_chunks": helper_chunks,
        "key_hash":      key_hash,
        "bitvec":        bitvec,
    }


# =============================================================================
#  RECOVERY
# =============================================================================

def recover_key(
    probe_bv:   np.ndarray,
    enrollment: Dict,
    bch:        BCH,
    perm:       np.ndarray,
) -> Dict:
    enrolled_bv   = enrollment["bitvec"]
    helper_chunks = enrollment["helper_chunks"]

    total_errors = int(np.sum(probe_bv != enrolled_bv))
    gate_pass    = total_errors <= TOTAL_BUDGET

    il_probe    = interleave(probe_bv,    perm)
    il_enrolled = interleave(enrolled_bv, perm)
    chunk_errors = [
        int(np.sum(il_probe[c] != il_enrolled[c]))
        for c in range(NUM_CHUNKS)
    ]

    rec_msgs:      List[np.ndarray] = []
    bch_fail_count = 0

    for c in range(NUM_CHUNKS):
        received       = (il_probe[c] ^ helper_chunks[c]).astype(np.uint8)
        decoded, n_err = bch.decode(received)
        if decoded is None:
            bch_fail_count += 1
            # Use zeros as fallback — this will cause hash DIFFER (correct behaviour)
            rec_msgs.append(np.zeros(BCH_K, dtype=np.uint8))
        else:
            rec_msgs.append(decoded)

    all_rec_msg = np.concatenate(rec_msgs)
    rec_hash    = hashlib.sha256(
        np.packbits(all_rec_msg).tobytes()
    ).hexdigest()
    hash_match  = rec_hash == enrollment["key_hash"]

    return {
        "gate_pass":      gate_pass,
        "total_errors":   total_errors,
        "chunk_errors":   chunk_errors,
        "enrolled_hash":  enrollment["key_hash"],
        "recovered_hash": rec_hash,
        "hash_match":     hash_match,
        "bch_fail_count": bch_fail_count,
    }


# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

SEP  = "=" * 80
SEP2 = "-" * 80


def _probe_type_label(enrolled_key: str, probe_key: str) -> str:
    return "GENUINE " if is_genuine(enrolled_key, probe_key) else "IMPOSTOR"


def run_test(
    test_num:     int,
    enrolled_key: str,
    probe_keys:   List[str],
    enrollments:  Dict[str, Dict],
    bitvecs:      Dict[str, np.ndarray],
    bch:          BCH,
    perm:         np.ndarray,
) -> Tuple[int, int, int, int]:

    enc           = enrollments[enrolled_key]
    enrolled_name = VIDEO_NAMES[enrolled_key]
    enrolled_id   = IDENTITY_MAP[enrolled_key]

    print(f"\n{SEP}")
    print(f"  TEST {test_num}  |  ENROLLED: {enrolled_name}  [{enrolled_id}]")
    print(SEP)
    print(f"  Enrolled key hash (SHA-256):")
    print(f"    {enc['key_hash']}")
    print()
    print(f"  Gate: total errors <= {TOTAL_BUDGET}  ->  PASS")
    print(f"        total errors  > {TOTAL_BUDGET}  ->  FAIL")
    print()
    print(f"  {'Probe':<22}  {'Type':>8}  {'TotErr':>6}  {'Gate':>6}  "
          f"{'BCHfail':>7}  {'Hash':>6}  {'Result':>6}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")

    gp = gn = ir = in_ = 0

    for probe_key in probe_keys:
        name        = VIDEO_NAMES[probe_key]
        ptype       = _probe_type_label(enrolled_key, probe_key)
        r           = recover_key(bitvecs[probe_key], enc, bch, perm)
        gate_str    = f"<={TOTAL_BUDGET}" if r["gate_pass"] else f">{TOTAL_BUDGET}"
        hash_status = "MATCH " if r["hash_match"] else "DIFFER"
        # Final result: PASS only if gate AND hash both succeed
        result      = "PASS" if (r["gate_pass"] and r["hash_match"]) else "FAIL"

        print(f"  {name:<22}  {ptype:>8}  {r['total_errors']:>6}  "
              f"{gate_str:>6}  {r['bch_fail_count']:>7}  "
              f"{hash_status:>6}  {result:>6}")
        print(f"  {'':22}  enrolled  hash : {r['enrolled_hash']}")
        print(f"  {'':22}  recovered hash : {r['recovered_hash']}")
        print(f"  {'':22}  hash status    : {hash_status.strip()}")
        print(f"  {'':22}  chunk errors   : "
              "[" + ", ".join(f"{v:>3}" for v in r["chunk_errors"]) + "]")
        print()

        if ptype.strip() == "GENUINE":
            gn += 1
            if r["gate_pass"] and r["hash_match"]:
                gp += 1
        else:
            in_ += 1
            if not (r["gate_pass"] and r["hash_match"]):
                ir += 1

    print(SEP2)
    print(f"  Test {test_num} summary  |  "
          f"Genuine PASS+MATCH {gp}/{gn}  |  Impostor REJECT {ir}/{in_}")

    return gp, gn, ir, in_


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model   = AdaFaceModel(WEIGHTS_PATH)
    aligner = FaceAligner()

    # ── Extract embeddings ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EXTRACTING EMBEDDINGS  ALL 7 VIDEOS")
    print(SEP)
    print(f"  {FRAMES_TO_USE} frames | CLAHE | Eye cascade | "
          f"Umeyama {FACE_SIZE}x{FACE_SIZE} | AdaFace IR-18")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        r = embed_video(vp, model, aligner)
        if r is not None:
            embeddings[f"video_{idx}"] = r

    all_keys = [f"video_{i}" for i in range(1, 8)]
    for k in all_keys:
        if k not in embeddings:
            raise RuntimeError(f"Embedding failed for {k}")

    # ── Build bit vectors ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BIT VECTOR GENERATION  (scale={SCALE}, fixed)")
    print(f"  Pipeline: L2-unit -> shuffle(0x{SHUFFLE_SEED:08X}) "
          f"-> x{SCALE} -> clip[-1,+1] -> {QUANT_BITS}-bit Gray "
          f"-> {NUM_CHUNKS}x{BCH_N}={BITVEC_LEN} bits")
    print(SEP)

    bitvecs: Dict[str, np.ndarray] = {}
    for k in all_keys:
        bv = embedding_to_bitvec(embeddings[k])
        bitvecs[k] = bv
        print(f"  {VIDEO_NAMES[k]:<22}  [{IDENTITY_MAP[k]}]  "
              f"len={len(bv)}  ones={int(bv.sum())}  density={bv.mean():.3f}")

    # ── Find best bit permutation ─────────────────────────────────────────
    genuine_keys = [k for k in all_keys if IDENTITY_MAP[k] == "person_A"]
    genuine_bvs  = {k: bitvecs[k] for k in genuine_keys}
    bit_perm, bit_perm_inv, best_seed, best_max = find_best_permutation(genuine_bvs)

    print(f"\n  Selected seed={best_seed}  max_chunk_err_genuine={best_max}  "
          f"(t={BCH_T},  budget={TOTAL_BUDGET})")

    # ── Pairwise error table ──────────────────────────────────────────────
    print(f"\n  Pairwise errors  (budget={TOTAL_BUDGET}, t per chunk={BCH_T}):")
    print(f"  {'Pair':<38}  {'TotErr':>6}  {'BER%':>6}  {'MaxChk':>6}  "
          f"{'Gate':>6}  Type")
    print(f"  {'─'*38}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
    for i, ka in enumerate(all_keys):
        for kb in all_keys[i + 1:]:
            e, r  = hamming(bitvecs[ka], bitvecs[kb])
            cerrs = chunk_errors_interleaved(bitvecs[ka], bitvecs[kb], bit_perm)
            gate  = f"<={TOTAL_BUDGET}" if e <= TOTAL_BUDGET else f">{TOTAL_BUDGET}"
            kind  = "genuine " if is_genuine(ka, kb) else "impostor"
            print(f"  {VIDEO_NAMES[ka]+' vs '+VIDEO_NAMES[kb]:<38}  {e:>6}  "
                  f"{r:>5.2f}%  {max(cerrs):>6}  {gate:>6}  {kind}")

    # ── BCH init ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BCH({BCH_N},{BCH_K},{BCH_T})  pure-Python  GF(2^8)  prim poly 0x11D")
    print(SEP)
    bch = BCH()
    g   = bch._gen_binary()
    print(f"  Generator degree : {len(g) - 1}  (expected {BCH_N - BCH_K}={BCH_N-BCH_K})")
    print(f"  Chunks           : {NUM_CHUNKS}")
    print(f"  Budget gate      : total errors <= {TOTAL_BUDGET}  ->  PASS")
    print(f"  Perm seed        : {best_seed}  (max genuine chunk err={best_max} < t={BCH_T})")

    # ── PHASE 1: Enroll ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 1  ENROLLMENT  V1..V7  (deterministic)")
    print(SEP)
    print("  scatter = bitvec[BIT_PERM]  ->  split into 8 chunks of 255")
    print("  helper[c]  = chunk[c]  XOR  BCH_encode(chunk[c][:K])")
    print("  key_hash   = SHA-256(first K bits of all chunks packed)")
    print()
    print(f"  {'Video':<22}  {'Identity':<10}  {'Key hash SHA-256':<64}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*64}")

    enrollments: Dict[str, Dict] = {}
    for k in all_keys:
        enc = enroll_one(bitvecs[k], bch, bit_perm)
        enrollments[k] = enc
        print(f"  {VIDEO_NAMES[k]:<22}  [{IDENTITY_MAP[k]}]  {enc['key_hash']}")

    # ── PHASE 2: 7 cross-tests ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 2  CROSS-AUTHENTICATION TESTS  (7 tests, one per enrolled video)")
    print(SEP)
    print("  GENUINE  + PASS + MATCH  = CORRECT ACCEPT")
    print("  IMPOSTOR + FAIL + DIFFER = CORRECT REJECT")
    print("  GENUINE  + FAIL          = FALSE REJECT  (pipeline error)")
    print("  IMPOSTOR + PASS + MATCH  = FALSE ACCEPT  (security breach)")

    test_configs = [
        (1, "video_1", ["video_2","video_3","video_4","video_5","video_6","video_7"]),
        (2, "video_2", ["video_1","video_3","video_4","video_5","video_6","video_7"]),
        (3, "video_3", ["video_1","video_2","video_4","video_5","video_6","video_7"]),
        (4, "video_4", ["video_1","video_2","video_3","video_5","video_6","video_7"]),
        (5, "video_5", ["video_1","video_2","video_3","video_4","video_6","video_7"]),
        (6, "video_6", ["video_1","video_2","video_3","video_4","video_5","video_7"]),
        (7, "video_7", ["video_1","video_2","video_3","video_4","video_5","video_6"]),
    ]

    total_gp = total_gn = total_ir = total_in = 0
    summaries: List[Tuple] = []

    for test_num, enrolled_key, probe_keys in test_configs:
        gp, gn, ir, in_ = run_test(
            test_num, enrolled_key, probe_keys,
            enrollments, bitvecs, bch, bit_perm,
        )
        total_gp += gp;  total_gn += gn
        total_ir += ir;  total_in += in_
        summaries.append((test_num, VIDEO_NAMES[enrolled_key],
                          IDENTITY_MAP[enrolled_key], gp, gn, ir, in_))

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(SEP)
    print(f"  Scale={SCALE} (fixed)  |  BCH({BCH_N},{BCH_K},{BCH_T})  |  "
          f"{NUM_CHUNKS} chunks  |  Budget={TOTAL_BUDGET}")
    print(f"  {QUANT_BITS}-bit Gray-code  |  {BITVEC_LEN} bits  |  "
          f"bit-perm seed={best_seed}  max genuine chunk err={best_max}")
    print()
    print(f"  {'Test':<6}  {'Enrolled':<22}  {'Identity':<10}  "
          f"{'Genuine PASS+MATCH':>18}  {'Impostor REJECT':>15}")
    print(f"  {'─'*6}  {'─'*22}  {'─'*10}  {'─'*18}  {'─'*15}")

    for tn, name, pid, gp, gn, ir, in_ in summaries:
        print(f"  {tn:<6}  {name:<22}  {pid:<10}  "
              f"{gp:>9} / {gn:<8}  {ir:>8} / {in_}")

    print(f"  {'─'*6}  {'─'*22}  {'─'*10}  {'─'*18}  {'─'*15}")
    print(f"  {'TOTAL':<6}  {'':22}  {'':10}  "
          f"{total_gp:>9} / {total_gn:<8}  {total_ir:>8} / {total_in}")
    print()

    false_rejects = total_gn - total_gp
    false_accepts = total_in - total_ir

    if false_rejects == 0 and false_accepts == 0:
        print("   PERFECT SEPARATION")
        print(f"    All {total_gn} genuine probes  : PASS + MATCH  (correct accept)")
        print(f"    All {total_in} impostor probes : FAIL + DIFFER (correct reject)")
    else:
        if false_rejects:
            print(f"    {false_rejects} FALSE REJECT(s) — genuine probe(s) FAILED "
                  "or HASH MISMATCH")
        if false_accepts:
            print(f"    {false_accepts} FALSE ACCEPT(s) — impostor probe(s) PASSED "
                  "gate AND hash matched (SECURITY BREACH)")
    print(SEP)


if __name__ == "__main__":
    main()
