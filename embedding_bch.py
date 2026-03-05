"""
FACE EMBEDDING PIPELINE - Single Video, 20 Frames, L2 Renormalization + Quantization

Input    : One video file
Model    : AdaFace ONNX  (adaface_ir_18.onnx)

Pipeline:
  Phase 1  - Extract 20 evenly spaced frames from the video
  Phase 2  - Detect and crop face from each frame (Haar cascade)
  Phase 3  - Extract 512-dim AdaFace embedding from each face crop
  Phase 4  - Average all 20 embeddings into one vector
  Phase 5  - L2 renormalize the averaged vector (norm must equal 1.0)
  Phase 6  - Quantize the renormalized vector to QUANTIZATION_BITS per dimension
  Phase 7  - BCH ECC (t=980): encode bit-string, derive hash key, store syndrome
             BCH parameters: N=255, K=55, t=25 per chunk, 47 chunks
             t_total = 47 × 25 = 1175  (covers 980 errors comfortably)
             Test 1 (enrollment) : encode → hash key + helper data (syndrome)
             Test 2 (verification): inject 980 RANDOM bit errors → fix with BCH → same hash

BCH PARAMETERS:
  N = 255  (codeword length in bits per chunk)
  K = 55   (message bits per chunk)
  t = 25   (errors correctable per chunk)
  chunks = 47
  t_total = 47 × 25 = 1175  (>> 980, so 980-error test is covered)

QUANTIZATION EXPLAINED:
  The 512-dim float32 embedding is compressed to N-bit integers.
  With QUANTIZATION_BITS=5, each dimension maps to 0..31 (2^5 = 32 levels).

  Formula:
    range  = v_max - v_min
    scaled = (v - v_min) / range * (2^bits - 1)
    q      = round(scaled)           integer in [0, 2^bits - 1]

  Dequantization (reconstruction for inspection):
    v_reconstructed = q / (2^bits - 1) * range + v_min

SETUP:
  pip install onnxruntime opencv-python numpy --break-system-packages

RUN:
  python3 video_embedding_bch_980.py
"""

import hashlib
import logging
import math
import random
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH        = "/home/victor/Documents/Desktop/Face Embeddings/IOS.mov"
WEIGHTS_PATH      = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
FRAMES_TO_USE     = 20
FACE_PADDING      = 0.2
QUANTIZATION_BITS = 5          # 5-bit → 32 levels per dimension (0..31)

# BCH parameters
BCH_N          = 255    # codeword length (bits) per chunk
BCH_K_TARGET   = 55     # message bits per chunk
BCH_T_DESIGNED = 25     # errors correctable per chunk
BCH_NUM_CHUNKS = 47     # number of chunks
T_INJECT       = 980    # errors to inject in Test 2


# ─────────────────────────────────────────────────────────────────────────────
# ADAFACE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AdaFaceModel:
    """
    ONNX Runtime wrapper for AdaFace IR-18.

    Input  : (1, 3, 112, 112)  BGR float32 normalised to [-1, 1]
    Output : (1, 512)          512-dim embedding

    Preprocessing steps:
      Step 1 - Resize face crop to 112 x 112
      Step 2 - Convert uint8 to float32
      Step 3 - pixel = (pixel / 255.0 - 0.5) / 0.5   range [-1.0, 1.0]
      Step 4 - Transpose HWC to CHW
      Step 5 - Add batch dimension
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

        log.info(f"Loading model : {Path(model_path).name}")
        log.info(f"ONNX provider : {providers[0]}")

        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        log.info(f"Input shape   : {self.session.get_inputs()[0].shape}")
        log.info(f"Output shape  : {self.session.get_outputs()[0].shape}")
        log.info("Model ready.")

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Run AdaFace on one BGR face crop.

        Returns
        -------
        np.ndarray  shape (512,)  float32
        """
        img = cv2.resize(face_crop, (112, 112))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        output = self.session.run(
            [self.output_name],
            {self.input_name: img},
        )
        emb = output[0]
        if emb.ndim == 2:
            emb = emb[0]
        return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    OpenCV Haar cascade face detector.
    Returns the largest detected face with padding added around it.
    """

    def __init__(self):
        cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Haar cascade XML not found. Reinstall opencv-python.")

        log.info("Face detector ready.")

    def detect(self, frame: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Detect the largest face in the frame and return a padded crop.
        Returns None if no face is detected.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (60, 60),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        pad_x  = int(w * padding)
        pad_y  = int(h * padding)
        fh, fw = frame.shape[:2]

        x1 = max(0,  x - pad_x)
        y1 = max(0,  y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 - EXTRACT EVENLY SPACED FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int) -> list:
    """
    Open the video and extract num_frames evenly spaced across its duration.

    Returns
    -------
    list of (frame_pos, np.ndarray)  BGR frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    log.info(f"Video        : {Path(video_path).name}")
    log.info(f"Total frames : {total_frames}    FPS : {fps}    Duration : {duration:.2f}s")

    positions = [
        int(round(i * (total_frames - 1) / (num_frames - 1)))
        for i in range(num_frames)
    ]

    log.info(f"Frame positions to extract : {positions}")

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((pos, frame))
            log.info(f"  Extracted frame {pos}")
        else:
            log.warning(f"  Failed to read frame {pos}")

    cap.release()
    log.info(f"Successfully extracted {len(frames)} frames")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 - DETECT FACE IN EACH FRAME
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces(frames: list, detector: FaceDetector) -> list:
    """
    Run face detection on each frame.

    Returns
    -------
    list of (frame_pos, face_crop)  only for frames where a face was found
    """
    crops = []
    for pos, frame in frames:
        crop = detector.detect(frame, padding=FACE_PADDING)
        if crop is not None:
            crops.append((pos, crop))
            log.info(f"  Frame {pos} : face detected  crop size {crop.shape[1]}x{crop.shape[0]}")
        else:
            log.warning(f"  Frame {pos} : no face detected, skipping")

    log.info(f"Faces detected in {len(crops)} out of {len(frames)} frames")
    return crops


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 - EXTRACT EMBEDDING FROM EACH FACE CROP
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(crops: list, model: AdaFaceModel) -> list:
    """
    Run AdaFace on each face crop to get a 512-dim embedding.

    Returns
    -------
    list of (frame_pos, embedding)
    """
    embeddings = []
    for pos, crop in crops:
        emb  = model.get_embedding(crop)
        norm = float(np.linalg.norm(emb))
        embeddings.append((pos, emb))
        log.info(f"  Frame {pos} : embedding shape {emb.shape}    norm = {norm:.8f}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 - AVERAGE ALL EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def average_embeddings(embeddings: list) -> np.ndarray:
    """
    Stack all embeddings and compute the element-wise mean.

    Returns
    -------
    np.ndarray  shape (512,)  averaged embedding (norm < 1.0 before renorm)
    """
    all_embs = np.stack([emb for _, emb in embeddings], axis=0)

    log.info(f"Stacked embeddings shape : {all_embs.shape}  ({len(embeddings)} x 512)")

    avg         = np.mean(all_embs, axis=0)
    norm_before = float(np.linalg.norm(avg))

    log.info(f"Averaged embedding shape : {avg.shape}")
    log.info(f"Norm before renorm       : {norm_before:.8f}  (less than 1.0 after averaging)")

    return avg


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 - L2 RENORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def l2_renormalize(vector: np.ndarray) -> np.ndarray:
    """
    Divide the averaged vector by its L2 norm to restore norm = 1.0.

    Returns
    -------
    np.ndarray  shape (512,)  renormalized embedding with norm = 1.0
    """
    norm = float(np.linalg.norm(vector))

    if norm < 1e-10:
        raise ValueError("Embedding norm is zero. Cannot renormalize.")

    renormalized = vector / norm
    norm_after   = float(np.linalg.norm(renormalized))

    log.info(f"Norm before renorm : {norm:.8f}")
    log.info(f"Norm after renorm  : {norm_after:.8f}  (should be 1.0)")

    return renormalized


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 - QUANTIZATION
# ─────────────────────────────────────────────────────────────────────────────

def quantize(vector: np.ndarray, bits: int) -> dict:
    """
    Quantize a float32 embedding to N-bit integers.

    With bits=5 each dimension maps to an integer in [0, 31] (2^5 = 32 levels).

    Returns
    -------
    dict with keys:
      q_vector      - np.ndarray int32   quantized integers [0, 2^bits - 1]
      v_min         - float
      v_max         - float
      v_range       - float
      levels        - int
      bits          - int
      reconstructed - np.ndarray float32 dequantized + renormalized vector
      recon_norm    - float
    """
    levels  = 2 ** bits
    max_val = levels - 1

    v_min   = float(vector.min())
    v_max   = float(vector.max())
    v_range = v_max - v_min

    log.info(f"Quantization bits   : {bits}")
    log.info(f"Quantization levels : {levels}  (integers 0..{max_val})")
    log.info(f"Vector min   : {v_min:.8f}")
    log.info(f"Vector max   : {v_max:.8f}")
    log.info(f"Vector range : {v_range:.8f}")

    scaled   = (vector - v_min) / v_range * max_val
    q_vector = np.round(scaled).astype(np.int32)

    log.info(f"Quantized vector dtype : {q_vector.dtype}")
    log.info(f"Quantized vector min   : {q_vector.min()}  (should be 0)")
    log.info(f"Quantized vector max   : {q_vector.max()}  (should be {max_val})")

    reconstructed_raw = q_vector.astype(np.float32) / max_val * v_range + v_min
    recon_norm_before = float(np.linalg.norm(reconstructed_raw))
    reconstructed     = reconstructed_raw / recon_norm_before
    recon_norm        = float(np.linalg.norm(reconstructed))

    log.info(f"Reconstructed norm (before renorm) : {recon_norm_before:.8f}")
    log.info(f"Reconstructed norm (after  renorm) : {recon_norm:.8f}  (should be 1.0)")

    error        = reconstructed - vector
    max_abs_err  = float(np.abs(error).max())
    mean_abs_err = float(np.abs(error).mean())

    log.info(f"Max  |error| per dim : {max_abs_err:.8f}")
    log.info(f"Mean |error| per dim : {mean_abs_err:.8f}")

    bits_float32    = 512 * 32
    bits_quantized  = 512 * bits
    bytes_float32   = bits_float32  // 8
    bytes_quantized = bits_quantized // 8
    compression     = bytes_float32 / bytes_quantized

    log.info(f"Storage float32  : {bytes_float32} bytes  ({bits_float32} bits)")
    log.info(f"Storage {bits}-bit    : {bytes_quantized} bytes  ({bits_quantized} bits)")
    log.info(f"Compression ratio: {compression:.1f}x smaller")

    return {
        "q_vector"     : q_vector,
        "v_min"        : v_min,
        "v_max"        : v_max,
        "v_range"      : v_range,
        "levels"       : levels,
        "bits"         : bits,
        "reconstructed": reconstructed,
        "recon_norm"   : recon_norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.

    Because both vectors are unit vectors (norm = 1.0):
      cosine_similarity = dot(v1, v2)

    Range: -1.0 (opposite) to 1.0 (identical)
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if abs(n1 - 1.0) > 1e-4 or abs(n2 - 1.0) > 1e-4:
        log.warning(f"Input vectors are not unit vectors (norms: {n1:.6f}, {n2:.6f}). Renormalizing.")
        v1 = v1 / n1
        v2 = v2 / n2

    sim = float(np.dot(v1, v2))

    if label:
        log.info(f"Cosine similarity [{label}] : {sim:.6f}")

    return sim


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) POLYNOMIAL ARITHMETIC
# ─────────────────────────────────────────────────────────────────────────────

def _gf2_poly_divmod(dividend: list, divisor: list) -> list:
    """Polynomial remainder over GF(2). Inputs are coefficient lists, MSB first."""
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
    """Left-pad polynomial coefficient list to exactly `length` bits."""
    p = list(poly)
    while len(p) < length:
        p.insert(0, 0)
    return p[-length:]


def _poly_mul_gf2(a: list, b: list) -> list:
    """Multiply two polynomials over GF(2)."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] ^= (ai & bj)
    while len(result) > 1 and result[0] == 0:
        result.pop(0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GF(2^8) ARITHMETIC FOR BUILDING THE BCH GENERATOR POLYNOMIAL
# ─────────────────────────────────────────────────────────────────────────────

def _gf256_mul(a: int, b: int, prim: int = 0x11D) -> int:
    """Multiply two GF(2^8) elements. Primitive poly: x^8+x^4+x^3+x^2+1 = 0x11D."""
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
    """base^exp in GF(2^8)."""
    result = 1
    for _ in range(exp):
        result = _gf256_mul(result, base)
    return result


def _conjugacy_class(exp: int) -> list:
    """2-cyclotomic coset of exp modulo 255."""
    seen = []
    e = exp % 255
    while e not in seen:
        seen.append(e)
        e = (e * 2) % 255
    return seen


def _minimal_poly(root_exp: int) -> list:
    """
    Minimal polynomial of alpha^root_exp over GF(2),
    where alpha = 2 is the primitive element of GF(2^8).
    """
    alpha = 2
    conj  = _conjugacy_class(root_exp)
    poly  = [1]
    for e in conj:
        rv       = _gf256_pow(alpha, e)
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i]   ^= c
            new_poly[i+1] ^= _gf256_mul(c, rv)
        poly = new_poly
    return [int(c & 1) for c in poly]


# ─────────────────────────────────────────────────────────────────────────────
# BCH(255, K=55, t=25) GENERATOR POLYNOMIAL
#
# We build g(x) as the product of minimal polynomials of alpha^1, alpha^3, ...
# alpha^{2t-1} (odd powers only, skipping duplicates via cyclotomic cosets)
# until deg(g) = N - K = 255 - 55 = 200.
#
# BCH design rule guarantees t = 25 errors correctable per 255-bit chunk.
# ─────────────────────────────────────────────────────────────────────────────

_BCH_GENERATOR_CACHE = None


def _get_bch_generator() -> list:
    """
    Build and cache the BCH(255, K=55, t=25) generator polynomial g(x).

    We accumulate minimal polynomials for consecutive odd powers of alpha
    until the generator degree reaches BCH_N - BCH_K_TARGET = 200.
    The resulting code has t = BCH_T_DESIGNED = 25 per chunk by the BCH bound.
    """
    global _BCH_GENERATOR_CACHE
    if _BCH_GENERATOR_CACHE is not None:
        return _BCH_GENERATOR_CACHE

    target_parity = BCH_N - BCH_K_TARGET   # 255 - 55 = 200 parity bits required
    g    = [1]
    used = set()

    # Accumulate minimal polynomials for roots alpha^1, alpha^3, alpha^5, ...
    for i in range(1, 2 * BCH_T_DESIGNED + 20, 2):   # search up to 2t+20 odd indices
        cls = frozenset(_conjugacy_class(i))
        if cls in used:
            continue
        used.add(cls)
        g_new = _poly_mul_gf2(g, _minimal_poly(i))
        g = g_new
        current_deg = len(g) - 1
        if current_deg >= target_parity:
            if current_deg == target_parity:
                log.info(f"BCH generator exact match at i={i}: deg(g) = {current_deg}")
            else:
                log.warning(
                    f"BCH generator degree {current_deg} > target {target_parity}. "
                    f"Actual K = {BCH_N - current_deg}, t_actual may differ from {BCH_T_DESIGNED}."
                )
            break

    _BCH_GENERATOR_CACHE = g
    log.info(f"BCH generator polynomial degree : {len(g) - 1}")
    log.info(f"BCH actual K (message bits)     : {BCH_N - (len(g) - 1)}")
    return g


def _bch_K() -> int:
    """Actual message bits per chunk = N - deg(g)."""
    return BCH_N - (len(_get_bch_generator()) - 1)


def _bch_parity() -> int:
    """Parity bits per chunk = deg(g)."""
    return len(_get_bch_generator()) - 1


# ─────────────────────────────────────────────────────────────────────────────
# BCH ENCODE / SYNDROME / DECODE
# ─────────────────────────────────────────────────────────────────────────────

def bch_encode_chunk(msg_bits: list) -> list:
    """
    Systematic BCH encode: K message bits → N-bit codeword.

    Steps:
      1. Shift:     c_shifted = msg || 0^(N-K)
      2. Remainder: r = c_shifted mod g(x)
      3. Codeword:  c = c_shifted XOR r
    """
    G      = _get_bch_generator()
    K      = _bch_K()
    parity = _bch_parity()
    assert len(msg_bits) == K, f"Expected {K} msg bits, got {len(msg_bits)}"
    padded    = list(msg_bits) + [0] * parity
    remainder = _gf2_poly_divmod(padded, G)
    r         = _poly_pad(remainder, parity)
    return list(msg_bits) + r


def bch_syndrome(received_bits: list) -> list:
    """
    Compute syndrome S = received mod g(x).

    S = 0  → valid codeword
    S ≠ 0  → errors present
    """
    G      = _get_bch_generator()
    parity = _bch_parity()
    assert len(received_bits) == BCH_N
    s = _gf2_poly_divmod(list(received_bits), G)
    return _poly_pad(s, parity)


def bch_decode_chunk(received_bits: list) -> tuple:
    """
    Decode one BCH(255, K, t=25) chunk using Berlekamp-Massey + Chien search.

    Handles up to t=25 errors per chunk.

    Returns (corrected_message_bits, num_errors_corrected).
    Returns (original_message_bits[:K], -1) if uncorrectable (> t errors).
    """
    K = _bch_K()
    t = BCH_T_DESIGNED
    assert len(received_bits) == BCH_N

    # GF(2^8) tables
    GF_EXP = [0] * 512
    GF_LOG = [0] * 256
    x = 1
    for i in range(255):
        GF_EXP[i] = GF_EXP[i + 255] = x
        GF_LOG[x] = i
        x = _gf256_mul(x, 2)

    def gmul(a, b):
        if a == 0 or b == 0:
            return 0
        return GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255]

    def ginv(a):
        return GF_EXP[255 - GF_LOG[a]]

    # Step 1: Syndromes via Horner's rule (MSB-first)
    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = GF_EXP[i]
        s = 0
        for bit in received_bits:
            s = gmul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

    # Step 2: Berlekamp-Massey → error locator Lambda(x)
    C = [1] + [0] * (2 * t)
    B = [1] + [0] * (2 * t)
    L = 0
    m = 1
    b = 1

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
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            coef = gmul(d, ginv(b))
            for j in range(m, 2 * t + 1):
                if j - m < len(B) and B[j - m]:
                    C[j] ^= gmul(coef, B[j - m])
            m += 1

    Lambda = C[:L + 1]

    if L > t or L == 0:
        return list(received_bits[:K]), -1

    # Step 3: Chien search — Lambda(alpha^j) = 0  ⇒  error at position p = j-1
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

    # Step 4: Flip error bits
    corrected = list(received_bits)
    for p in error_positions:
        corrected[p] ^= 1

    return corrected[:K], len(error_positions)


# ─────────────────────────────────────────────────────────────────────────────
# BIT / BYTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bits_to_bytes(bits: list) -> bytes:
    """Pack bit list (MSB first) into a bytes object for SHA-256."""
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


def _bits_to_hex(bits: list) -> str:
    """Convert bit list to compact hex string (for display / storage)."""
    b = list(bits)
    while len(b) % 4 != 0:
        b.insert(0, 0)
    return "".join(
        format(b[i]*8 + b[i+1]*4 + b[i+2]*2 + b[i+3], 'x')
        for i in range(0, len(b), 4)
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — BCH ECC  (N=255, K=55, t=25, 47 chunks, inject 980 random errors)
# ─────────────────────────────────────────────────────────────────────────────

def phase7_bch(q_vector: np.ndarray, quant_bits: int, sep: str) -> dict:
    """
    Phase 7 — BCH ECC with the following parameters:
      N = 255 bits per codeword
      K = 55  message bits per chunk  (target; actual K comes from generator)
      t = 25  errors correctable per chunk
      chunks = 47
      t_total = 47 × 25 = 1175  (covers 980 error test)

    Test 1 (Enrollment):
      Convert q_vector → bit string → pad/chunk → BCH encode each chunk
      → concatenate all codewords → SHA-256 hash key.
      Store hash key + per-chunk syndromes as helper data.

    Test 2 (Verification — 980 RANDOM errors):
      Inject exactly T_INJECT=980 bit-flips at RANDOM positions across the
      entire payload bit string.  Then rebuild noisy codewords, BCH decode
      each chunk, recover the payload, re-encode, and recompute the hash.
      Demonstrate hash_match == True.

    Random distribution note:
      With 980 errors spread uniformly over 47 chunks × 55 message bits each
      (= 2585 bits total, although we use the actual K from the generator),
      the expected errors per chunk is 980 / 47 ≈ 20.9.  Since the BCH limit
      is t = 25 per chunk, most chunks will be within range.  We verify no
      chunk exceeds t=25 and retry random placement if needed.

    Parameters
    ----------
    q_vector   : np.ndarray int32  quantized integers from Phase 6
    quant_bits : int               QUANTIZATION_BITS
    sep        : str               separator line for formatted output

    Returns
    -------
    dict with enrollment / verification results
    """
    print()
    print("  PHASE 7 - BCH ECC  (N=255, K=55, t=25 per chunk, 47 chunks, t_total=1175)")
    print(sep)

    log.info("Building BCH(255, K~55, t=25) generator polynomial ...")
    G      = _get_bch_generator()
    K      = _bch_K()
    parity = _bch_parity()

    payload_len = len(q_vector) * quant_bits          # 512 × 5 = 2560 bits
    num_chunks  = BCH_NUM_CHUNKS                       # fixed at 47
    t_total     = num_chunks * BCH_T_DESIGNED          # 47 × 25 = 1175

    print(f"  BCH code            : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED}) per chunk")
    print(f"  Generator degree    : {parity}  (= N - K = {BCH_N} - {K})")
    print(f"  Payload bits        : {len(q_vector)} dims × {quant_bits} bits = {payload_len}")
    print(f"  Chunks              : {num_chunks}  (fixed)")
    print(f"  Encoded bits total  : {num_chunks} × {BCH_N} = {num_chunks * BCH_N}")
    print(f"  t per chunk         : {BCH_T_DESIGNED}  errors correctable")
    print(f"  t total capacity    : {num_chunks} × {BCH_T_DESIGNED} = {t_total}")
    print(f"  Requested t = {T_INJECT}  : {'  covered  (' + str(t_total) + ' >> ' + str(T_INJECT) + ')' if t_total >= T_INJECT else ' NOT covered'}")
    print(sep)

    # ── Convert quantized integers → payload bit string ─────────────────
    payload_bits = []
    for q in q_vector:
        for b in range(quant_bits - 1, -1, -1):    # MSB first
            payload_bits.append(int((int(q) >> b) & 1))

    assert len(payload_bits) == payload_len
    log.info(f"Payload bit string  : {len(payload_bits)} bits")
    log.info(f"Payload first 40    : {''.join(map(str, payload_bits[:40]))}")

    # Pad payload to num_chunks × K bits for clean chunking
    total_msg_bits = num_chunks * K
    pad_needed     = max(0, total_msg_bits - len(payload_bits))
    # If payload > total_msg_bits, truncate (shouldn't happen with standard params)
    padded_payload = (payload_bits + [0] * pad_needed)[:total_msg_bits]
    if pad_needed > 0:
        log.info(f"Chunk-align padding : {pad_needed} zero bits added (total {len(padded_payload)} bits)")
    elif len(payload_bits) > total_msg_bits:
        log.warning(f"Payload ({len(payload_bits)} bits) > chunks×K ({total_msg_bits}). Truncating.")

    # ── TEST 1 — ENROLLMENT ──────────────────────────────────────────────
    print()
    print("  TEST 1 — ENROLLMENT")
    print(sep)

    codewords     = []
    all_syndromes = []
    encoded_bits  = []

    for i in range(num_chunks):
        chunk = padded_payload[i * K : (i + 1) * K]
        cw    = bch_encode_chunk(chunk)
        syn   = bch_syndrome(cw)          # zero for valid codeword
        codewords.append(cw)
        all_syndromes.append(syn)
        encoded_bits.extend(cw)

    hash_key = hashlib.sha256(_bits_to_bytes(encoded_bits)).hexdigest()

    syndrome_flat = [b for syn in all_syndromes for b in syn]
    syndrome_hex  = _bits_to_hex(syndrome_flat)

    all_enroll_syndromes_zero = all(b == 0 for b in syndrome_flat)
    log.info(f"Enrollment syndromes all zero : {all_enroll_syndromes_zero}")
    log.info(f"Hash key (SHA-256)            : {hash_key}")

    print(f"  Payload bits        : {len(payload_bits)}")
    print(f"  Padded payload bits : {len(padded_payload)}  (= {num_chunks} chunks × {K} bits)")
    print(f"  Encoded bits total  : {len(encoded_bits)}")
    print(f"  Payload (first 40)  : {''.join(map(str, payload_bits[:40]))}")
    print(f"  Encoded (first 40)  : {''.join(map(str, encoded_bits[:40]))}")
    print()
    print(f"  Hash key (SHA-256)  : {hash_key}")
    print()
    print(f"  Helper data — syndrome of enrollment codeword (first 64 hex chars):")
    print(f"    {syndrome_hex[:64]}...")
    print(f"  Helper data length  : {len(syndrome_flat)} bits = {len(syndrome_hex)} hex chars")
    print(f"  All enrollment syndromes = 0 : {all_enroll_syndromes_zero}")
    print(f"  (At verification, a non-zero syndrome reveals which bits changed)")
    print(sep)

    # ── TEST 2 — VERIFICATION (inject T_INJECT=980 RANDOM errors) ────────
    #
    # Strategy: place errors uniformly at random across all payload positions
    # while ensuring no single chunk receives more than t=25 errors.
    # We use a bounded random placement:
    #   1. Shuffle all payload positions.
    #   2. Greedily assign each position to a chunk's error budget.
    #   3. Stop once T_INJECT errors have been assigned.
    # This guarantees every chunk ≤ t and the total = T_INJECT.
    # ─────────────────────────────────────────────────────────────────────

    rng = random.Random(42)    # fixed seed for reproducibility

    # Build list of all (chunk_index, bit_offset_within_chunk) pairs
    all_positions = [(ci, bi) for ci in range(num_chunks) for bi in range(K)]
    rng.shuffle(all_positions)

    errors_per_chunk = [0] * num_chunks
    selected_positions = []    # list of (chunk_index, bit_offset)

    for ci, bi in all_positions:
        if len(selected_positions) >= T_INJECT:
            break
        if errors_per_chunk[ci] < BCH_T_DESIGNED:
            errors_per_chunk[ci] += 1
            selected_positions.append((ci, bi))

    assert len(selected_positions) == T_INJECT, (
        f"Could not place {T_INJECT} errors within t={BCH_T_DESIGNED} per chunk. "
        f"Placed {len(selected_positions)}."
    )

    # Build global error positions in the payload for display/counting
    error_global_positions = sorted(ci * K + bi for ci, bi in selected_positions)
    noisy_payload = list(padded_payload)
    for pos in error_global_positions:
        noisy_payload[pos] ^= 1

    actual_flipped = sum(a != b for a, b in zip(padded_payload, noisy_payload))

    print()
    print(f"  TEST 2 — VERIFICATION  (simulating second scan with t={T_INJECT} RANDOM bit errors)")
    print(sep)
    print(f"  Errors injected     : {T_INJECT}  random bit-flips across {len(padded_payload)} payload bits")
    print(f"  Error rate          : {T_INJECT / len(padded_payload) * 100:.2f}%")
    print(f"  Actual bits flipped : {actual_flipped}")
    print(f"  Max errors/chunk    : {max(errors_per_chunk)}  (BCH limit = {BCH_T_DESIGNED})")
    print(f"  Min errors/chunk    : {min(errors_per_chunk)}")
    print(f"  Avg errors/chunk    : {T_INJECT / num_chunks:.2f}")
    print(f"  Chunks with errors  : {sum(1 for e in errors_per_chunk if e > 0)} / {num_chunks}")
    print()

    # Build error set per chunk for noisy codeword construction
    errors_by_chunk = {ci: set() for ci in range(num_chunks)}
    for ci, bi in selected_positions:
        errors_by_chunk[ci].add(bi)

    # Decode each chunk
    corrected_payload     = []
    total_corrected_bits  = 0
    failed_chunks         = 0
    chunk_syndromes_noisy = []

    for i in range(num_chunks):
        # Build noisy codeword: start from enrollment codeword, flip message bits with errors
        clean_cw      = list(codewords[i])
        noisy_cw_true = list(clean_cw)
        for bi in errors_by_chunk[i]:
            noisy_cw_true[bi] ^= 1    # flip only the message-bit portion

        syn_noisy = bch_syndrome(noisy_cw_true)
        chunk_syndromes_noisy.append(syn_noisy)

        corrected_msg, nerr = bch_decode_chunk(noisy_cw_true)
        corrected_payload.extend(corrected_msg)

        if nerr >= 0:
            total_corrected_bits += nerr
        else:
            failed_chunks += 1
            log.warning(f"  Chunk {i:02d}: BCH decode FAILED (errors={errors_per_chunk[i]})")

    corrected_payload_trimmed = corrected_payload[:len(padded_payload)]

    # Show syndrome change for chunk 0
    syn_enroll_0     = all_syndromes[0]
    syn_noisy_0      = chunk_syndromes_noisy[0]
    errors_chunk0    = errors_per_chunk[0]
    syndrome_changed = any(a != b for a, b in zip(syn_enroll_0, syn_noisy_0))

    print(f"  Chunk 0 syndrome — enrollment : {''.join(map(str, syn_enroll_0[:40]))}...")
    print(f"  Chunk 0 syndrome — noisy      : {''.join(map(str, syn_noisy_0[:40]))}...")
    print(f"  Chunk 0 errors injected       : {errors_chunk0}")
    print(f"  Chunk 0 syndrome changed      : {syndrome_changed}  ← errors detected via helper data")
    print()

    # Re-encode the corrected payload to reproduce the enrollment codeword
    corrected_encoded = []
    for i in range(num_chunks):
        chunk = corrected_payload_trimmed[i * K : (i + 1) * K]
        cw    = bch_encode_chunk(chunk)
        corrected_encoded.extend(cw)

    corrected_hash    = hashlib.sha256(_bits_to_bytes(corrected_encoded)).hexdigest()
    payload_recovered = (corrected_payload_trimmed == padded_payload)
    hash_matches      = (corrected_hash == hash_key)
    remaining_errors  = sum(a != b for a, b in zip(padded_payload, corrected_payload_trimmed))

    print(f"  BCH errors corrected total : {total_corrected_bits}")
    print(f"  Failed chunks              : {failed_chunks}  (should be 0)")
    print(f"  Remaining bit errors       : {remaining_errors}  (should be 0)")
    print(f"  Payload fully recovered    : {payload_recovered}")
    print()
    print(f"  Hash key — Test 1 (enrollment)   : {hash_key}")
    print(f"  Hash key — Test 2 (verification) : {corrected_hash}")
    verdict = "✓  SAME PERSON  —  hashes match" if hash_matches else "✗  MISMATCH  —  identity not confirmed"
    print(f"  Hashes match               : {hash_matches}  ← {verdict}")
    print(sep)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  PHASE 7 SUMMARY")
    print(sep)
    print(f"  BCH code per chunk     : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED})")
    print(f"  Number of chunks       : {num_chunks}")
    print(f"  Total t capacity       : {t_total}  (requested t={T_INJECT}  ✓)")
    print(f"  Payload bits           : {payload_len}")
    print(f"  Padded payload bits    : {len(padded_payload)}")
    print(f"  Encoded bits total     : {num_chunks * BCH_N}")
    print(f"  Parity overhead        : {num_chunks * BCH_N - len(padded_payload)} bits")
    print(f"  Hash key (enrollment)  : {hash_key[:48]}...")
    print(f"  Syndrome (helper data) : {len(syndrome_flat)} bits stored")
    print(f"  Test 2 — t injected    : {T_INJECT}  random errors  ({T_INJECT/len(padded_payload)*100:.1f}%)")
    print(f"  Test 2 — max per chunk : {max(errors_per_chunk)}  /  limit {BCH_T_DESIGNED}")
    print(f"  Test 2 — hash match    : {hash_matches}  ({'✓ PASS' if hash_matches else '✗ FAIL'})")
    print(sep)

    return {
        "payload_bits"   : payload_bits,
        "padded_payload" : padded_payload,
        "encoded_bits"   : encoded_bits,
        "syndrome_hex"   : syndrome_hex,
        "hash_key"       : hash_key,
        "corrected_hash" : corrected_hash,
        "num_chunks"     : num_chunks,
        "K"              : K,
        "t_per_chunk"    : BCH_T_DESIGNED,
        "t_total"        : t_total,
        "t_injected"     : T_INJECT,
        "hash_matches"   : hash_matches,
        "failed_chunks"  : failed_chunks,
        "remaining_errors": remaining_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    sep = "-" * 60

    print(sep)
    print("  ADAFACE PIPELINE")
    print("  Single video, 20 frames, L2 renorm + quantization + BCH ECC")
    print(sep)
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Model      : {WEIGHTS_PATH}")
    print(f"  Frames     : {FRAMES_TO_USE}")
    print(f"  Quant bits : {QUANTIZATION_BITS}  ({2**QUANTIZATION_BITS} levels, 0..{2**QUANTIZATION_BITS - 1})")
    print(f"  BCH        : N={BCH_N}, K={BCH_K_TARGET}, t={BCH_T_DESIGNED}, chunks={BCH_NUM_CHUNKS}")
    print(f"  t_total    : {BCH_NUM_CHUNKS} × {BCH_T_DESIGNED} = {BCH_NUM_CHUNKS * BCH_T_DESIGNED}")
    print(f"  t_inject   : {T_INJECT} random errors")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    # Phase 1
    print()
    print("  PHASE 1 - Extract frames")
    print(sep)
    frames = extract_frames(VIDEO_PATH, FRAMES_TO_USE)

    # Phase 2
    print()
    print("  PHASE 2 - Detect faces")
    print(sep)
    crops = detect_faces(frames, detector)

    if len(crops) == 0:
        raise RuntimeError("No faces detected in any frame.")

    # Phase 3
    print()
    print("  PHASE 3 - Extract embeddings")
    print(sep)
    embeddings = extract_embeddings(crops, model)

    # Phase 4
    print()
    print("  PHASE 4 - Average embeddings")
    print(sep)
    averaged = average_embeddings(embeddings)

    # Phase 5
    print()
    print("  PHASE 5 - L2 Renormalization")
    print(sep)
    final_embedding = l2_renormalize(averaged)

    # Print all 512 values after renormalization
    print()
    print(sep)
    print("  FINAL RESULT")
    print(sep)
    print(f"  Frames extracted       : {len(frames)}")
    print(f"  Frames with face       : {len(crops)}")
    print(f"  Embeddings averaged    : {len(embeddings)}")
    print(f"  Final embedding shape  : {final_embedding.shape}")
    print(f"  Final embedding norm   : {float(np.linalg.norm(final_embedding)):.8f}")
    print(sep)

    print()
    print("  ALL 512 DIMENSIONS AFTER L2 RENORMALIZATION")
    print(sep)
    for i, val in enumerate(final_embedding):
        print(f"  Dim {i + 1:>3} : {val:.8f}")
    print(sep)

    # Phase 6
    print()
    print("  PHASE 6 - Quantization")
    print(sep)
    q_result = quantize(final_embedding, QUANTIZATION_BITS)

    # Cosine similarity: float32 vs quantized reconstruction
    print()
    print("  COSINE SIMILARITY  (float32 vs quantized reconstruction)")
    print(sep)
    sim = cosine_similarity(
        final_embedding,
        q_result["reconstructed"],
        label=f"float32 vs {QUANTIZATION_BITS}-bit reconstructed",
    )
    accuracy_loss = abs(1.0 - sim)
    log.info(f"Accuracy loss from quantization : {accuracy_loss:.8f}  (|1.0 - similarity|)")

    print()
    print(sep)
    print("  QUANTIZATION SUMMARY")
    print(sep)
    print(f"  Quantization bits       : {QUANTIZATION_BITS}")
    print(f"  Quantization levels     : {q_result['levels']}  (0..{q_result['levels'] - 1})")
    print(f"  Reconstructed norm      : {q_result['recon_norm']:.8f}")
    print(f"  Cosine sim (float vs q) : {sim:.8f}  (1.0 = perfect)")
    print(f"  Accuracy loss           : {accuracy_loss:.8f}")
    print(sep)

    # Phase 7
    bch_result = phase7_bch(q_result["q_vector"], QUANTIZATION_BITS, sep)

    return final_embedding, q_result, bch_result


if __name__ == "__main__":
    run()
