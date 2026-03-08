"""
AdaFace + BCH Fuzzy Commitment Pipeline
=========================================

EMBEDDING PIPELINE (unchanged from phase_pipeline.py)
  1. Extract top-20 sharpest frames (scan 60 candidates)
  2. Detect face + eye cascade → Umeyama 112x112 alignment
  3. AdaFace IR-18 → raw 512-dim embedding
  4. L2 normalise → unit vector
  5. Average all frame unit vectors → L2 renormalise → final unit embedding

BCH FUZZY COMMITMENT
  Binarisation : sign(v_i) → 1 if v_i > 0 else 0  → 512-bit payload
  BCH params   : BCH(255, 71, t=28)  x  8 chunks   → 8 × 255 = 2040 bits codeword
  Enrollment   : helper = codeword XOR padded_payload  (2040 bits, stored)
  Key          : HMAC-SHA256(seed, payload_bytes) → 32-byte H key  (stored)
  Recovery     : (helper XOR probe_bits_padded) → BCH decode → recovered_payload
               → HMAC-SHA256(seed, recovered) → compare with H_key → ACCEPT/REJECT

BCH PHASE 1 — Enroll V1, V2, V3, V4 separately
  Each video → own (helper_data, H_key, seed)

BCH PHASE 2 — Cross-tests
  Test 1 : Use V1 helper_data → probe with V2,V3,V4 (genuine)  V5,V6,V7 (impostor)
  Test 2 : Use V2 helper_data → probe with V3,V4 (genuine)     V5,V6,V7 (impostor)
  Test 3 : Use V3 helper_data → probe with V4 (genuine)        V5,V6,V7 (impostor)
"""

import hashlib
import hmac
import logging
import secrets
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

GENUINE_SET   = {"video_1", "video_2", "video_3", "video_4"}
IMPOSTOR_SET  = {"video_5", "video_6", "video_7"}
FRAMES_TO_USE = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112

REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

# BCH(255, 71, t=28) x 8 chunks
# 8 x 71 = 568 payload capacity; we use first 512 (sign bits)
# 8 x 255 = 2040 bits codeword
BCH_N       = 255
BCH_K       = 71
BCH_T       = 28
BCH_CHUNKS  = 8
PAYLOAD_BITS = 512


# =============================================================================
#  SECTION 1 — FACE ALIGNER
# =============================================================================

class FaceAligner:
    def __init__(self):
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        if self.face_cascade.empty():
            raise RuntimeError("Face Haar cascade not found.")
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.eye_ok = not self.eye_cascade.empty()
        self.clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        log.info(f"FaceAligner ready  |  eye cascade: {'yes' if self.eye_ok else 'no (geometry fallback)'}")

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _detect_face(self, gray):
        for sf, mn, ms in [(1.05, 6, 60), (1.05, 3, 40), (1.10, 2, 30)]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
            if len(faces) > 0:
                return tuple(max(faces, key=lambda f: f[2]*f[3]))
        return None

    def _detect_eyes(self, gray, fx, fy, fw, fh):
        if not self.eye_ok:
            return None
        roi  = gray[fy: fy + int(fh*0.60), fx: fx + fw]
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20))
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.10, minNeighbors=2, minSize=(15, 15))
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        centres = sorted(
            [np.array([fx + ex + ew//2, fy + ey + eh//2], dtype=np.float32)
             for ex, ey, ew, eh in eyes],
            key=lambda p: p[0])
        return centres[0], centres[1]

    @staticmethod
    def _landmarks(x, y, w, h, le=None, re=None):
        le = le if le is not None else np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        re = re if re is not None else np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        return np.array([
            le, re,
            [x+0.50*w, y+0.57*h],
            [x+0.35*w, y+0.76*h],
            [x+0.65*w, y+0.76*h],
        ], dtype=np.float32)

    @staticmethod
    def _umeyama(src, dst):
        n    = src.shape[0]
        mu_s = src.mean(0); mu_d = dst.mean(0)
        sc   = src - mu_s;  dc   = dst - mu_d
        vs   = (sc**2).sum() / n
        if vs < 1e-10:
            return None
        cov  = (dc.T @ sc) / n
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

    def align(self, frame):
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


# =============================================================================
#  SECTION 2 — ADAFACE MODEL
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
        log.info(f"AdaFace IR-18 loaded | {providers[0]}")

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
#  SECTION 3 — FRAME EXTRACTION
# =============================================================================

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


# =============================================================================
#  SECTION 4 — EMBED ONE VIDEO → UNIT EMBEDDING
# =============================================================================

def embed_video(video_path: str, model: AdaFaceModel, aligner: FaceAligner) -> Optional[np.ndarray]:
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# =============================================================================
#  SECTION 5 — BINARISATION
#
#  sign(v_i) → 1 if v_i > 0, else 0  →  512-bit payload
#  Standard approach for BCH fuzzy commitment on L2-normalised embeddings.
#  No quantisation noise. Each bit directly represents the sign of a dimension.
# =============================================================================

def binarise(embedding: np.ndarray) -> np.ndarray:
    """L2-normalised 512-dim float32 → 512-bit uint8 array (sign binarisation)."""
    return (embedding > 0).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack 512-bit uint8 array → 64 bytes (MSB first within each byte)."""
    assert len(bits) == 512
    out = bytearray(64)
    for i in range(64):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i*8 + j])
        out[i] = byte
    return bytes(out)


# =============================================================================
#  SECTION 6 — BCH CODEC
#
#  Uses bchlib: BCH(t=28, m=8) → GF(2^8), N=255, K=71
#
#  Chunk layout (per chunk, 64 payload bits → 255-bit codeword):
#    msg (9 bytes = 72 bits): bit[0]=0 (padding), bits[1..64] = payload bits
#    ecc (23 bytes = 184 bits): BCH parity
#    codeword = first 255 bits of (msg + ecc)
#
#  8 chunks × 64 payload bits = 512 bits total payload
#  8 chunks × 255 codeword bits = 2040 bits total codeword
# =============================================================================

def _get_bch():
    try:
        import bchlib
        return bchlib.BCH(BCH_T, m=8)
    except Exception as e:
        raise RuntimeError(
            f"bchlib unavailable: {e}\n"
            f"Install with: pip install bchlib"
        )


def _pad_payload_to_codeword(payload_bits: np.ndarray) -> np.ndarray:
    """
    Lay 512 payload bits into a 2040-bit array matching the codeword structure.
    Each chunk uses 64 payload bits placed at positions [1..64] within the
    9-byte (72-bit) message field, which starts at offset chunk*255 in the codeword.
    """
    padded = np.zeros(BCH_N * BCH_CHUNKS, dtype=np.uint8)
    for chunk_idx in range(BCH_CHUNKS):
        start_payload = chunk_idx * 64
        start_cw      = chunk_idx * BCH_N
        # bit 0 of the 9-byte message is always 0 (padding)
        # bits 1..64 carry the payload
        padded[start_cw + 1 : start_cw + 65] = payload_bits[start_payload : start_payload + 64]
    return padded


def bch_encode_512(payload_bits: np.ndarray) -> np.ndarray:
    """
    512-bit payload → 2040-bit BCH codeword (8 x BCH(255,71,28)).
    Returns uint8 array of length 2040.
    """
    bch    = _get_bch()
    result = []

    for chunk_idx in range(BCH_CHUNKS):
        # Build 72-bit message: bit 0 = 0, bits 1..64 = payload chunk
        chunk_bits = np.zeros(72, dtype=np.uint8)
        chunk_bits[1:65] = payload_bits[chunk_idx * 64 : chunk_idx * 64 + 64]

        msg_bytes = bytearray(9)
        for i in range(9):
            b = 0
            for j in range(8):
                b = (b << 1) | int(chunk_bits[i*8 + j])
            msg_bytes[i] = b

        ecc = bch.encode(bytes(msg_bytes))                # 23 bytes parity

        codeword_bytes = bytes(msg_bytes) + ecc           # 32 bytes = 256 bits
        cw_bits = np.unpackbits(np.frombuffer(codeword_bytes, dtype=np.uint8))
        result.append(cw_bits[:BCH_N])                    # first 255 bits

    return np.concatenate(result)                         # 2040 bits


def bch_decode_512(received_bits: np.ndarray) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    2040-bit received word → decoded 512-bit payload (or None if uncorrectable).
    Returns (payload_bits_512, errors_per_chunk).
    Negative error count means uncorrectable chunk.
    """
    bch         = _get_bch()
    recovered   = []
    errors_list = []
    success     = True

    for chunk_idx in range(BCH_CHUNKS):
        cw_bits = received_bits[chunk_idx * BCH_N : (chunk_idx + 1) * BCH_N]

        # Pad to 256 bits for bchlib
        padded   = np.zeros(256, dtype=np.uint8)
        padded[:BCH_N] = cw_bits
        cw_bytes = np.packbits(padded).tobytes()          # 32 bytes

        msg_bytes = bytearray(cw_bytes[:9])
        ecc_bytes = bytearray(cw_bytes[9:32])             # 23 bytes

        try:
            nerr = bch.decode(bytes(msg_bytes), bytes(ecc_bytes))
            if nerr < 0:
                errors_list.append(-1)
                success = False
                recovered.append(np.zeros(64, dtype=np.uint8))
                continue

            bch.correct(msg_bytes, ecc_bytes)
            errors_list.append(nerr)

            msg_bits = np.unpackbits(np.frombuffer(bytes(msg_bytes), dtype=np.uint8))
            recovered.append(msg_bits[1:65])              # bits 1..64 = payload

        except Exception as exc:
            log.debug(f"BCH decode chunk {chunk_idx}: {exc}")
            errors_list.append(-1)
            success = False
            recovered.append(np.zeros(64, dtype=np.uint8))

    payload_512 = np.concatenate(recovered) if success else None
    return payload_512, errors_list


# =============================================================================
#  SECTION 7 — FUZZY COMMITMENT  (Juels-Wattenberg)
#
#  ENROLLMENT:
#    payload       = binarise(embedding)                     512 bits
#    codeword      = BCH_encode(payload)                    2040 bits
#    padded_payload = _pad_payload_to_codeword(payload)     2040 bits
#    helper_data   = codeword XOR padded_payload            2040 bits  ← stored
#    seed          = secrets.token_bytes(32)                  32 bytes  ← stored
#    h_key         = HMAC-SHA256(seed, payload_bytes)         32 bytes  ← stored
#
#  AUTHENTICATION:
#    probe_bits    = binarise(probe_embedding)               512 bits
#    padded_probe  = _pad_payload_to_codeword(probe_bits)   2040 bits
#    received      = helper_data XOR padded_probe           2040 bits
#    recovered, errors = BCH_decode(received)
#    if all chunks OK:
#        h_recovered = HMAC-SHA256(seed, recovered_bytes)
#        ACCEPT iff h_recovered == h_key
#    else:
#        REJECT
# =============================================================================

def enroll(embedding: np.ndarray) -> Dict:
    """
    Enroll one embedding.

    Returns:
      payload_bits  : np.ndarray (512,)   sign binarisation of embedding
      codeword_bits : np.ndarray (2040,)  BCH codeword
      helper_data   : np.ndarray (2040,)  codeword XOR padded_payload  [store this]
      seed          : bytes (32)          random HMAC seed              [store this]
      h_key         : bytes (32)          HMAC-SHA256(seed, payload)    [store this]
    """
    payload_bits   = binarise(embedding)
    codeword_bits  = bch_encode_512(payload_bits)
    padded_payload = _pad_payload_to_codeword(payload_bits)
    helper_data    = (codeword_bits ^ padded_payload).astype(np.uint8)
    seed           = secrets.token_bytes(32)
    payload_bytes  = bits_to_bytes(payload_bits)
    h_key          = hmac.new(seed, payload_bytes, hashlib.sha256).digest()

    return {
        "payload_bits":  payload_bits,
        "codeword_bits": codeword_bits,
        "helper_data":   helper_data,
        "seed":          seed,
        "h_key":         h_key,
    }


def authenticate(probe_embedding: np.ndarray, enrollment: Dict) -> Dict:
    """
    Attempt to authenticate probe_embedding against an enrollment record.

    Returns:
      decision         : bool    True = ACCEPT, False = REJECT
      errors_per_chunk : list    BCH errors per chunk (negative = uncorrectable)
      total_errors     : int
      uncorrectable    : bool
      bit_error_rate   : float   BER% between probe payload and enrolled payload
    """
    probe_bits    = binarise(probe_embedding)
    padded_probe  = _pad_payload_to_codeword(probe_bits)
    received_bits = (enrollment["helper_data"] ^ padded_probe).astype(np.uint8)

    recovered_payload, errors_per_chunk = bch_decode_512(received_bits)

    uncorrectable = (recovered_payload is None or any(e < 0 for e in errors_per_chunk))

    if uncorrectable:
        # BER still measurable even if BCH fails
        n_err = int(np.sum(probe_bits != enrollment["payload_bits"]))
        return {
            "decision":         False,
            "errors_per_chunk": errors_per_chunk,
            "total_errors":     sum(e for e in errors_per_chunk if e >= 0),
            "uncorrectable":    True,
            "bit_error_rate":   n_err / PAYLOAD_BITS * 100.0,
        }

    recovered_bytes = bits_to_bytes(recovered_payload)
    h_recovered     = hmac.new(enrollment["seed"], recovered_bytes, hashlib.sha256).digest()
    accepted        = hmac.compare_digest(h_recovered, enrollment["h_key"])

    n_err = int(np.sum(probe_bits != enrollment["payload_bits"]))
    return {
        "decision":         accepted,
        "errors_per_chunk": errors_per_chunk,
        "total_errors":     sum(errors_per_chunk),
        "uncorrectable":    False,
        "bit_error_rate":   n_err / PAYLOAD_BITS * 100.0,
    }


# =============================================================================
#  HELPERS
# =============================================================================

SEP  = "=" * 72
SEP2 = "-" * 72


def _probe_type(enroll_key: str, probe_key: str) -> str:
    if enroll_key in GENUINE_SET and probe_key in GENUINE_SET:
        return "GENUINE "
    return "IMPOSTOR"


def _decision_label(result: Dict) -> str:
    if result["uncorrectable"]:
        return "REJECT (uncorrectable)"
    return "ACCEPT [OK]" if result["decision"] else "REJECT [FAIL]"


# =============================================================================
#  MAIN
# =============================================================================

def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model   = AdaFaceModel(WEIGHTS_PATH)
    aligner = FaceAligner()

    # ── Extract embeddings — all 7 videos ─────────────────────────────────
    print(f"\n{SEP}")
    print("  EXTRACTING EMBEDDINGS — ALL 7 VIDEOS")
    print(SEP)
    print(f"  {FRAMES_TO_USE} frames | CLAHE | Eye cascade | Umeyama 112x112")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        key    = f"video_{idx}"
        result = embed_video(vp, model, aligner)
        if result is not None:
            embeddings[key] = result

    # ── Cosine similarity reference table ─────────────────────────────────
    print(f"\n{SEP}")
    print("  COSINE SIMILARITY — ALL PAIRS  (reference)")
    print(SEP)
    print(f"\n  {'Pair':<38}  {'Similarity':>10}")
    print(f"  {'─'*38}  {'─'*10}")
    all_keys = [f"video_{i}" for i in range(1, 8) if f"video_{i}" in embeddings]
    for i, ka in enumerate(all_keys):
        for kb in all_keys[i+1:]:
            sim = cosine_sim(embeddings[ka], embeddings[kb])
            na  = VIDEO_NAMES.get(ka, ka)
            nb  = VIDEO_NAMES.get(kb, kb)
            print(f"  {na + ' vs ' + nb:<38}  {sim:>10.4f}")

    # =========================================================================
    #  BCH PHASE 1 — ENROLL V1, V2, V3, V4 SEPARATELY
    # =========================================================================
    print(f"\n{SEP}")
    print("  BCH PHASE 1 — ENROLL V1, V2, V3, V4 SEPARATELY")
    print(SEP)
    print(f"  Scheme  : Juels-Wattenberg fuzzy commitment")
    print(f"  BCH     : BCH({BCH_N},{BCH_K},t={BCH_T})  x  {BCH_CHUNKS} chunks")
    print(f"  Payload : sign(embedding)  ->  {PAYLOAD_BITS} bits  (512 sign bits)")
    print(f"  Codeword: {BCH_N * BCH_CHUNKS} bits  ({BCH_CHUNKS} x {BCH_N})")
    print(f"  H_key   : HMAC-SHA256(seed, payload_bytes)  ->  32 bytes")
    print()
    print(f"  {'Video':<20}  {'Payload bits [0:8]':>20}  {'H_key (first 16 hex chars)':>28}")
    print(f"  {'─'*20}  {'─'*20}  {'─'*28}")

    enrollments: Dict[str, Dict] = {}
    enroll_keys = ["video_1", "video_2", "video_3", "video_4"]

    for k in enroll_keys:
        if k not in embeddings:
            raise RuntimeError(f"Missing embedding for {k}")
        enc = enroll(embeddings[k])
        enrollments[k] = enc
        name = VIDEO_NAMES[k]
        pbits = enc["payload_bits"][:8].tolist()
        hkey  = enc["h_key"].hex()[:16]
        print(f"  {name:<20}  {str(pbits):>20}  {hkey:>28}")

    print()
    print(f"  Each enrollment uses a unique random seed -> unique H_key per session.")
    print(f"  Helper data = 2040 bits stored per enrolled video.")

    # =========================================================================
    #  BCH PHASE 2 — CROSS AUTHENTICATION TESTS
    #  Test 1: V1 helper  <- probe V2,V3,V4,V5,V6,V7
    #  Test 2: V2 helper  <- probe V3,V4,V5,V6,V7
    #  Test 3: V3 helper  <- probe V4,V5,V6,V7
    # =========================================================================

    test_configs = [
        ("video_1", ["video_2","video_3","video_4","video_5","video_6","video_7"]),
        ("video_2", ["video_3","video_4","video_5","video_6","video_7"]),
        ("video_3", ["video_4","video_5","video_6","video_7"]),
    ]

    genuine_decisions  = []
    impostor_decisions = []

    for test_num, (enrolled_key, probe_keys) in enumerate(test_configs, start=1):
        enrolled_name = VIDEO_NAMES[enrolled_key]

        print(f"\n{SEP}")
        print(f"  BCH PHASE 2 — TEST {test_num}  |  ENROLLED: {enrolled_name}")
        print(SEP)
        print(f"  Using helper_data + seed + H_key from {enrolled_name} enrollment")
        print(f"  BCH corrects up to t={BCH_T} bit errors per {BCH_N}-bit chunk")
        print()
        print(f"  {'Probe':<20}  {'Type':>8}  {'BER%':>6}  {'TotErr':>6}  {'Errors per chunk (8 chunks)':>34}  {'Decision'}")
        print(f"  {'─'*20}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*34}  {'─'*22}")

        enc = enrollments[enrolled_key]

        for probe_key in probe_keys:
            if probe_key not in embeddings:
                print(f"  {VIDEO_NAMES.get(probe_key,probe_key):<20}  [embedding missing — skipped]")
                continue

            probe_name = VIDEO_NAMES[probe_key]
            result     = authenticate(embeddings[probe_key], enc)
            ptype      = _probe_type(enrolled_key, probe_key)
            dlabel     = _decision_label(result)

            ber_str  = f"{result['bit_error_rate']:.1f}%"
            terr_str = str(result["total_errors"])

            ec = result["errors_per_chunk"]
            ec_str = "[" + " ".join(
                f"{e:2d}" if e >= 0 else "--" for e in ec
            ) + "]"

            print(f"  {probe_name:<20}  {ptype:>8}  {ber_str:>6}  {terr_str:>6}  {ec_str:>34}  {dlabel}")

            if ptype == "GENUINE ":
                genuine_decisions.append(result["decision"])
            else:
                impostor_decisions.append(result["decision"])

    # =========================================================================
    #  FINAL SUMMARY
    # =========================================================================
    print(f"\n{SEP}")
    print("  BCH PIPELINE — FINAL SUMMARY")
    print(SEP)
    print()
    print(f"  BCH scheme     : BCH({BCH_N},{BCH_K},t={BCH_T})  x {BCH_CHUNKS} chunks")
    print(f"  Payload        : {PAYLOAD_BITS} sign bits  ({PAYLOAD_BITS//8} bytes)")
    print(f"  Codeword       : {BCH_N*BCH_CHUNKS} bits  ({BCH_N*BCH_CHUNKS//8} bytes)")
    print(f"  Max correctable: t={BCH_T} errors / {BCH_N}-bit chunk = {BCH_T/BCH_N*100:.1f}% per chunk")
    print()

    total_genuine  = len(genuine_decisions)
    total_impostor = len(impostor_decisions)

    if total_genuine > 0:
        n_accept = genuine_decisions.count(True)
        n_reject = genuine_decisions.count(False)
        tar = n_accept / total_genuine * 100.0
        frr = n_reject / total_genuine * 100.0
        print(f"  Genuine probes   : {total_genuine}")
        print(f"    ACCEPT  (TAR)  : {n_accept:>2}  ({tar:.1f}%)")
        print(f"    REJECT  (FRR)  : {n_reject:>2}  ({frr:.1f}%)")

    print()

    if total_impostor > 0:
        n_accept = impostor_decisions.count(True)
        n_reject = impostor_decisions.count(False)
        far = n_accept / total_impostor * 100.0
        tnr = n_reject / total_impostor * 100.0
        print(f"  Impostor probes  : {total_impostor}")
        print(f"    ACCEPT  (FAR)  : {n_accept:>2}  ({far:.1f}%)")
        print(f"    REJECT  (TNR)  : {n_reject:>2}  ({tnr:.1f}%)")

    print()

    all_genuine_ok      = all(genuine_decisions)  if genuine_decisions  else False
    all_impostor_blocked = not any(impostor_decisions) if impostor_decisions else False

    if all_genuine_ok and all_impostor_blocked:
        verdict = "PERFECT SEPARATION  -- All genuine ACCEPT, all impostor REJECT"
    elif all_impostor_blocked:
        n_fail = genuine_decisions.count(False)
        verdict = f"PARTIAL  -- All impostor REJECT, but {n_fail} genuine(s) REJECT (FRR issue)"
    elif all_genuine_ok:
        n_leak = impostor_decisions.count(True)
        verdict = f"PARTIAL  -- All genuine ACCEPT, but {n_leak} impostor(s) ACCEPT (FAR issue)"
    else:
        verdict = "FAILURE  -- Check BER gap and BCH parameters"

    print(f"  Verdict : {verdict}")
    print()
    print(f"  Expected genuine BER   < {BCH_T/BCH_N*100:.1f}% per chunk  ->  correctable  ->  ACCEPT")
    print(f"  Expected impostor BER  > {BCH_T/BCH_N*100:.1f}% per chunk  ->  uncorrectable ->  REJECT")
    print(SEP)


if __name__ == "__main__":
    main()
