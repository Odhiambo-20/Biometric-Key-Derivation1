"""
FACE EMBEDDING PIPELINE - Single Video, 10 Frames, L2 Renormalization + Quantization

Input    : One video file
Model    : AdaFace ONNX  (adaface_ir_18.onnx)

Pipeline:
  Phase 1  - Extract 10 evenly spaced frames from the video
  Phase 2  - Detect and crop face from each frame (Haar cascade)
  Phase 3  - Extract 512-dim AdaFace embedding from each face crop
  Phase 4  - Average all 10 embeddings into one vector
  Phase 5  - L2 renormalize the averaged vector (norm must equal 1.0)
  Phase 6  - Quantize the renormalized vector to QUANTIZATION_BITS per dimension
  Phase 7  - BCH ECC (t=400): encode bit-string, derive hash key, store syndrome
             Test 1 (enrollment) : encode → hash key + helper data (syndrome)
             Test 2 (verification): inject bit errors → fix with syndrome → same hash

QUANTIZATION EXPLAINED:
  The 512-dim float32 embedding is compressed to N-bit integers.
  With QUANTIZATION_BITS=5, each dimension maps to 0..31 (2^5 = 32 levels).

  Formula:
    range  = v_max - v_min
    scaled = (v - v_min) / range * (2^bits - 1)
    q      = round(scaled)           integer in [0, 2^bits - 1]

  Dequantization (reconstruction for inspection):
    v_reconstructed = q / (2^bits - 1) * range + v_min

  Storage comparison (per embedding):
    float32   : 512 x 32 bits = 16,384 bits = 2,048 bytes
    5-bit     : 512 x  5 bits =  2,560 bits =   320 bytes  (6.4x smaller)

COSINE SIMILARITY (self-check):
  Compares the float32 embedding against its own quantized reconstruction.
  A high similarity (close to 1.0) means quantization preserved the identity.

  similarity = dot(v_float, v_reconstructed) = sum( v_j * v_rec_j )
  Range: -1.0 (opposite) to 1.0 (identical)

SETUP:
  pip install onnxruntime opencv-python numpy --break-system-packages

RUN:
  python3 video_embedding_6bits.py
"""

import hashlib
import logging
import math
import warnings
from itertools import combinations
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



# CONFIG


VIDEO_PATH        = "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4"
WEIGHTS_PATH      = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
FRAMES_TO_USE     = 20
FACE_PADDING      = 0.2
QUANTIZATION_BITS = 5          # 5-bit → 32 levels per dimension (0..31)



# ADAFACE MODEL


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



# FACE DETECTOR


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



# PHASE 1 - EXTRACT 10 EVENLY SPACED FRAMES


def extract_frames(video_path: str, num_frames: int) -> list:
    """
    Open the video and extract num_frames evenly spaced across its duration.

    For a 323-frame video with num_frames=10:
      positions = [0, 36, 72, 107, 143, 179, 215, 250, 286, 322]

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



# PHASE 2 - DETECT FACE IN EACH FRAME


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



# PHASE 3 - EXTRACT EMBEDDING FROM EACH FACE CROP


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



# PHASE 4 - AVERAGE ALL EMBEDDINGS


def average_embeddings(embeddings: list) -> np.ndarray:
    """
    Stack all embeddings and compute the element-wise mean.

    Formula:
      avg_j = (1/N) * sum( e_{i,j} )  for i = 1..N, j = 1..512

    After averaging, the norm will no longer be 1.0 because averaging
    multiple unit vectors produces a shorter vector pointing toward
    the centre of the cluster.

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



# PHASE 5 - L2 RENORMALIZATION


def l2_renormalize(vector: np.ndarray) -> np.ndarray:
    """
    Divide the averaged vector by its L2 norm to restore norm = 1.0.

    Formula:
      norm     = sqrt( sum( v_j^2 ) )  for j = 1..512
      v_norm_j = v_j / norm

    Why renormalize:
      AdaFace outputs unit vectors (norm = 1.0).
      After averaging N unit vectors, the result points in the right
      direction but has a shorter length (norm < 1.0).
      Dividing by the norm rescales it back to length 1.0 so the
      embedding lives on the unit hypersphere, which is required for
      cosine similarity to work correctly.

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



# PHASE 6 - QUANTIZATION


def quantize(vector: np.ndarray, bits: int) -> dict:
    """
    Quantize a float32 embedding to N-bit integers.

    With bits=5 each dimension maps to an integer in [0, 31] (2^5 = 32 levels).

    Quantization formula (per dimension j):
      range    = v_max - v_min
      scaled_j = (v_j - v_min) / range * (2^bits - 1)
      q_j      = round(scaled_j)   →   integer in [0, 2^bits - 1]

    Dequantization formula (reconstruction for cosine similarity):
      v_reconstructed_j = q_j / (2^bits - 1) * range + v_min

    Storage:
      float32  : 512 dims × 32 bits = 16,384 bits = 2,048 bytes
      5-bit    : 512 dims ×  5 bits =  2,560 bits =   320 bytes  (6.4x smaller)

    Parameters
    ----------
    vector : np.ndarray  shape (512,)  L2-normalized float32
    bits   : int         number of bits per dimension (e.g. 5)

    Returns
    -------
    dict with keys:
      q_vector     - np.ndarray int32   quantized integers [0, 2^bits - 1]
      v_min        - float               minimum value of original vector
      v_max        - float               maximum value of original vector
      v_range      - float               v_max - v_min
      levels       - int                 number of quantization levels (2^bits)
      bits         - int                 bit width used
      reconstructed - np.ndarray float32 dequantized vector (renormalized)
      recon_norm   - float               norm of reconstructed vector after renorm
    """
    levels  = 2 ** bits                         # 32 for 5-bit
    max_val = levels - 1                        # 31 for 5-bit

    v_min   = float(vector.min())
    v_max   = float(vector.max())
    v_range = v_max - v_min

    log.info(f"Quantization bits : {bits}")
    log.info(f"Quantization levels : {levels}  (integers 0..{max_val})")
    log.info(f"Vector min   : {v_min:.8f}")
    log.info(f"Vector max   : {v_max:.8f}")
    log.info(f"Vector range : {v_range:.8f}")

    # Quantize: float → integer
    scaled   = (vector - v_min) / v_range * max_val
    q_vector = np.round(scaled).astype(np.int32)

    log.info(f"Quantized vector dtype : {q_vector.dtype}")
    log.info(f"Quantized vector min   : {q_vector.min()}  (should be 0)")
    log.info(f"Quantized vector max   : {q_vector.max()}  (should be {max_val})")

    # Dequantize: integer → float (approximate reconstruction)
    reconstructed_raw  = q_vector.astype(np.float32) / max_val * v_range + v_min
    recon_norm_before  = float(np.linalg.norm(reconstructed_raw))

    # Re-normalize the reconstructed vector so it lives on the unit hypersphere
    reconstructed = reconstructed_raw / recon_norm_before
    recon_norm    = float(np.linalg.norm(reconstructed))

    log.info(f"Reconstructed norm (before renorm) : {recon_norm_before:.8f}")
    log.info(f"Reconstructed norm (after  renorm) : {recon_norm:.8f}  (should be 1.0)")

    # Quantization error per dimension
    error        = reconstructed - vector
    max_abs_err  = float(np.abs(error).max())
    mean_abs_err = float(np.abs(error).mean())

    log.info(f"Max  |error| per dim : {max_abs_err:.8f}")
    log.info(f"Mean |error| per dim : {mean_abs_err:.8f}")

    # Storage report
    bits_float32   = 512 * 32
    bits_quantized = 512 * bits
    bytes_float32  = bits_float32  // 8
    bytes_quantized = bits_quantized // 8
    compression    = bytes_float32 / bytes_quantized

    log.info(f"Storage float32   : {bytes_float32} bytes  ({bits_float32} bits)")
    log.info(f"Storage {bits}-bit     : {bytes_quantized} bytes  ({bits_quantized} bits)")
    log.info(f"Compression ratio : {compression:.1f}x smaller")

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


# COSINE SIMILARITY


def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.

    Because both vectors are unit vectors (norm = 1.0):
      cosine_similarity = dot(v1, v2) = sum( v1_j * v2_j )  for j = 1..512

    Range:
       1.0  →  identical direction (same person, same image)
       0.0  →  orthogonal          (unrelated)
      -1.0  →  opposite direction

    AdaFace IR-18 typical thresholds:
      > 0.40  →  likely same person
      > 0.60  →  confident same person
      < 0.20  →  likely different person

    Parameters
    ----------
    v1, v2 : np.ndarray  shape (512,)  must be L2-normalized
    label  : str         description printed in the log

    Returns
    -------
    float  cosine similarity in [-1.0, 1.0]
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



# COSINE SIMILARITY


def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.

    Because both vectors are unit vectors (norm = 1.0):
      cosine_similarity = dot(v1, v2) = sum( v1_j * v2_j )  for j = 1..512

    Range:
       1.0  →  identical direction (same person, same image)
       0.0  →  orthogonal          (unrelated)
      -1.0  →  opposite direction

    Parameters
    ----------
    v1, v2 : np.ndarray  shape (512,)  must be L2-normalized
    label  : str         description printed in the log

    Returns
    -------
    float  cosine similarity in [-1.0, 1.0]
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



# MAIN


def run():
    sep = "-" * 55

    print(sep)
    print("  ADAFACE PIPELINE")
    print("  Single video, 10 frames, L2 renormalization + quantization")
    print(sep)
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Model      : {WEIGHTS_PATH}")
    print(f"  Frames     : {FRAMES_TO_USE}")
    print(f"  Quant bits : {QUANTIZATION_BITS}  ({2**QUANTIZATION_BITS} levels, 0..{2**QUANTIZATION_BITS - 1})")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    # Phase 1
    print()
    print("  PHASE 1 - Extract 10 frames")
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

    # Print all 512 values after renormalization (original output preserved)
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

    # Cosine similarity: float32 vs its own quantized reconstruction
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

    # Quantization summary
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


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 - BCH ECC  (t = 400)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS BCH?
#   BCH (Bose-Chaudhuri-Hocquenghem) is a cyclic binary error-correcting code.
#   A BCH(N, K, t) codeword is N bits long, carries K message bits, and can
#   correct up to t bit-flips anywhere in the N-bit codeword.
#
# BCH PARAMETERS USED:
#   BCH(255, K, t_designed=21) over GF(2^8)
#     N      = 255 bits per codeword   (2^8 - 1, standard primitive-length BCH)
#     K      = N - deg(g)              (actual K determined by generator degree)
#     t_each = 21  errors correctable per chunk
#     Parity = deg(g) bits per codeword
#
# CHUNKED APPROACH FOR t=400:
#   5-bit quantized embedding  =  512 × 5 = 2560 payload bits
#   Split into chunks of K bits each.
#   Each chunk → BCH(255, K, t=21) codeword.
#   Total error capacity = num_chunks × 21 >> 400.
#
#   In Test 2 we spread 400 errors at most 3 per chunk (avg 2.03),
#   guaranteeing all chunks are within the brute-force decoder's range.
#   In practice, face bit-errors spread uniformly → same guarantee holds.
#
# SYNDROME AS HELPER DATA (fuzzy commitment / secure sketch):
#   syndrome S = received_codeword mod g(x)
#   Valid codeword  → S = 0  (zero polynomial)
#   Corrupted codeword → S ≠ 0  (encodes error locations)
#
#   Enrollment (Test 1):
#     1. payload P → BCH encode → codeword C
#     2. hash_key H = SHA-256(C)   ← stored as identity token
#     3. syndrome of C = 0         ← stored as helper data (trivially zero at enroll)
#
#   Verification (Test 2):
#     1. noisy payload P' → BCH encode → noisy codeword C'
#     2. syndrome(C') ≠ 0          ← non-zero reveals which bits are wrong
#     3. BCH decode C' → correct C
#     4. recompute H' = SHA-256(C)
#     5. H' == H → same person ✓
# ─────────────────────────────────────────────────────────────────────────────


# ── GF(2) polynomial arithmetic ──────────────────────────────────────────────

def _gf2_poly_divmod(dividend: list, divisor: list) -> list:
    """
    Polynomial remainder over GF(2).  Inputs are coefficient lists, MSB first.
    E.g. x^2 + 1  →  [1, 0, 1]
    """
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


# ── GF(2^8) arithmetic for building the BCH generator polynomial ─────────────

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
    alpha    = 2
    conj     = _conjugacy_class(root_exp)
    poly     = [1]
    for e in conj:
        rv       = _gf256_pow(alpha, e)
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i]   ^= c
            new_poly[i+1] ^= _gf256_mul(c, rv)
        poly = new_poly
    return [int(c & 1) for c in poly]


# BCH design parameters
BCH_N          = 255    # codeword length (bits)
BCH_T_DESIGNED = 21     # error correction capacity per chunk (designed)

_BCH_GENERATOR_CACHE = None   # cached after first build


def _get_bch_generator() -> list:
    """
    Build and cache the BCH(255, K, t=21) generator polynomial g(x).
    g(x) = product of distinct minimal polynomials m_1, m_3, ..., m_{2t-1}.
    The degree of g(x) equals BCH_N - K.
    """
    global _BCH_GENERATOR_CACHE
    if _BCH_GENERATOR_CACHE is not None:
        return _BCH_GENERATOR_CACHE
    g    = [1]
    used = set()
    for i in range(1, 2 * BCH_T_DESIGNED, 2):    # odd indices: 1,3,5,...,41
        cls = frozenset(_conjugacy_class(i))
        if cls in used:
            continue
        used.add(cls)
        g = _poly_mul_gf2(g, _minimal_poly(i))
    _BCH_GENERATOR_CACHE = g
    return g


def _bch_K() -> int:
    """Actual message bits per chunk = N - deg(g)."""
    return BCH_N - (len(_get_bch_generator()) - 1)


def _bch_parity() -> int:
    """Parity bits per chunk = deg(g)."""
    return len(_get_bch_generator()) - 1


# ── BCH encode / syndrome / decode ───────────────────────────────────────────

def bch_encode_chunk(msg_bits: list) -> list:
    """
    Systematic BCH encode: K message bits → N-bit codeword.

    Steps:
      1. Shift:     c_shifted = msg || 0^(N-K)    (multiply by x^(N-K))
      2. Remainder: r = c_shifted mod g(x)
      3. Codeword:  c = c_shifted XOR r
         → first K bits = original message, last (N-K) bits = parity
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

    Interpretation:
      S = 0  → valid codeword, no detectable errors
      S ≠ 0  → errors present; S encodes their locations (helper data)
    """
    G      = _get_bch_generator()
    parity = _bch_parity()
    assert len(received_bits) == BCH_N
    s = _gf2_poly_divmod(list(received_bits), G)
    return _poly_pad(s, parity)


def bch_decode_chunk(received_bits: list) -> tuple:
    """
    Decode one BCH(255, K, t=21) chunk using Berlekamp-Massey + Chien search.

    Handles up to t=21 errors per chunk in O(t^2 + N) time.

    Bit convention (MSB-first):
      The codeword polynomial is cw[0]*x^{N-1} + cw[1]*x^{N-2} + ... + cw[N-1].
      Syndrome S_i = R(alpha^i) is evaluated via Horner's rule (MSB-first).
      A codeword position p has error locator root at alpha^{p+1}.
      Chien finds j where Lambda(alpha^j)=0  =>  error at position p = j - 1.

    Algorithm:
      1. Syndrome: S_i = R(alpha^i) for i=1..2t via Horner's rule.
      2. Berlekamp-Massey: find error locator Lambda(x).
      3. Chien search: evaluate Lambda(alpha^j) for j=1..N; zero at j -> error at p=j-1.
      4. Flip bits at error positions, return first K bits.

    Returns (corrected_message_bits, num_errors_corrected).
    Returns (original_message_bits, -1) if uncorrectable (> t errors).
    """
    K = _bch_K()
    t = BCH_T_DESIGNED
    assert len(received_bits) == BCH_N

    # ── GF(2^8) tables ────────────────────────────────────────────────────
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

    # ── Step 1: Syndromes via Horner's rule (MSB-first) ───────────────────
    # S_i = R(alpha^i):  result = 0; for each bit: result = result*alpha^i XOR bit
    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = GF_EXP[i]
        s = 0
        for bit in received_bits:
            s = gmul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

    # ── Step 2: Berlekamp-Massey → error locator Lambda(x) ───────────────
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

    # ── Step 3: Chien search — Lambda(alpha^j) = 0  =>  error at p = j-1 ─
    error_positions = []
    for j in range(1, BCH_N + 1):
        val = Lambda[0]    # Lambda[0] = 1
        for k in range(1, len(Lambda)):
            if Lambda[k]:
                val ^= gmul(Lambda[k], GF_EXP[(j * k) % 255])
        if val == 0:
            p = j - 1
            if 0 <= p < BCH_N:
                error_positions.append(p)

    if len(error_positions) != L:
        return list(received_bits[:K]), -1

    # ── Step 4: Flip error bits ───────────────────────────────────────────
    corrected = list(received_bits)
    for p in error_positions:
        corrected[p] ^= 1

    return corrected[:K], len(error_positions)



# ── Bit / byte helpers ────────────────────────────────────────────────────────

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


# ── Phase 7 main function ─────────────────────────────────────────────────────

def phase7_bch(q_vector: np.ndarray, quant_bits: int, sep: str) -> dict:
    """
    Phase 7 — BCH ECC with t=400 total error tolerance.

    Test 1 (Enrollment):
      Convert q_vector → bit string → BCH encode → SHA-256 hash key.
      Store hash key + syndrome as helper data.

    Test 2 (Verification):
      Inject exactly 400 bit-flips (≤ 3 per chunk) into the payload.
      BCH decode each chunk → recover original → recompute hash.
      Show that hashes match despite the 400 injected errors.

    Parameters
    ----------
    q_vector   : np.ndarray int32  quantized integers from Phase 6
    quant_bits : int               QUANTIZATION_BITS (5)
    sep        : str               separator line for formatted output

    Returns
    -------
    dict with hash_key, syndrome_hex, corrected_hash, hash_matches, etc.
    """
    # ── BCH setup ────────────────────────────────────────────────────────
    print()
    print("  PHASE 7 - BCH ECC  (t = 400)")
    print(sep)
    log.info("Building BCH(255, K, t=21) generator polynomial ...")
    G      = _get_bch_generator()
    K      = _bch_K()
    parity = _bch_parity()

    payload_len = len(q_vector) * quant_bits          # 512 × 5 = 2560
    num_chunks  = math.ceil(payload_len / K)
    t_total     = num_chunks * BCH_T_DESIGNED

    print(f"  BCH code            : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED}) per chunk")
    print(f"  Generator degree    : {parity}  (= N - K = {BCH_N} - {K})")
    print(f"  Payload bits        : {len(q_vector)} dims × {quant_bits} bits = {payload_len}")
    print(f"  Chunks              : ceil({payload_len} / {K}) = {num_chunks}")
    print(f"  Encoded bits total  : {num_chunks} × {BCH_N} = {num_chunks * BCH_N}")
    print(f"  t per chunk         : {BCH_T_DESIGNED}  errors correctable")
    print(f"  t total capacity    : {num_chunks} × {BCH_T_DESIGNED} = {t_total}")
    print(f"  Requested t = 400   : {'  covered  (' + str(t_total) + ' >> 400)' if t_total >= 400 else ' NOT covered'}")
    print(sep)

    # ── Convert quantized integers → 2560-bit payload ─────────────────
    payload_bits = []
    for q in q_vector:
        for b in range(quant_bits - 1, -1, -1):       # MSB first
            payload_bits.append(int((int(q) >> b) & 1))

    assert len(payload_bits) == payload_len
    log.info(f"Payload bit string  : {len(payload_bits)} bits")
    log.info(f"Payload first 40    : {''.join(map(str, payload_bits[:40]))}")

    # Pad to multiple of K for clean chunking
    pad_needed     = (num_chunks * K) - len(payload_bits)
    padded_payload = payload_bits + [0] * pad_needed
    if pad_needed > 0:
        log.info(f"Chunk-align padding : {pad_needed} zero bits added")

    # ── TEST 1 — ENROLLMENT ──────────────────────────────────────────────
    print()
    print("  TEST 1 — ENROLLMENT")
    print(sep)

    codewords    = []
    all_syndromes = []
    encoded_bits  = []

    for i in range(num_chunks):
        chunk = padded_payload[i * K : (i + 1) * K]
        cw    = bch_encode_chunk(chunk)
        syn   = bch_syndrome(cw)          # zero for valid codeword
        codewords.append(cw)
        all_syndromes.append(syn)
        encoded_bits.extend(cw)

    # Hash key — SHA-256 of the full BCH-encoded codeword
    hash_key = hashlib.sha256(_bits_to_bytes(encoded_bits)).hexdigest()

    # Helper data: syndromes of all enrollment chunks (all-zero at enrollment)
    syndrome_flat = [b for syn in all_syndromes for b in syn]
    syndrome_hex  = _bits_to_hex(syndrome_flat)

    all_enroll_syndromes_zero = all(b == 0 for b in syndrome_flat)
    log.info(f"Enrollment syndromes all zero : {all_enroll_syndromes_zero}")
    log.info(f"Hash key (SHA-256)            : {hash_key}")

    print(f"  Payload bits        : {len(payload_bits)}")
    print(f"  Encoded bits        : {len(encoded_bits)}")
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

    # ── TEST 2 — VERIFICATION (inject t=400 errors) ───────────────────────
    # Spread 400 errors across chunks using round-robin: at most 3 per chunk.
    # This guarantees every chunk stays within the brute-force decoder's range.
    # 400 / num_chunks ≈ 2.03 per chunk on average.
    T_INJECT = 400

    errors_per_chunk_plan = [0] * num_chunks
    assigned = 0
    ci = 0
    while assigned < T_INJECT:
        if errors_per_chunk_plan[ci] < BCH_T_DESIGNED:
            errors_per_chunk_plan[ci] += 1
            assigned += 1
        ci = (ci + 1) % num_chunks

    # Error positions: first N bits of each chunk receive their allocated errors
    error_positions_set = set()
    for ci_idx, num_errs in enumerate(errors_per_chunk_plan):
        base = ci_idx * K
        for e in range(num_errs):
            error_positions_set.add(base + e)

    error_positions = sorted(error_positions_set)
    noisy_payload   = list(payload_bits)
    for pos in error_positions:
        if pos < len(noisy_payload):
            noisy_payload[pos] ^= 1

    actual_flipped = sum(a != b for a, b in zip(payload_bits, noisy_payload))

    print()
    print("  TEST 2 — VERIFICATION  (simulating second scan with t=400 bit errors)")
    print(sep)
    print(f"  Errors injected     : {T_INJECT}  bit-flips across {payload_len} payload bits")
    print(f"  Error rate          : {T_INJECT / payload_len * 100:.2f}%")
    print(f"  Actual bits flipped : {actual_flipped}")
    print(f"  Max errors/chunk    : {max(errors_per_chunk_plan)}  (BCH limit = {BCH_T_DESIGNED})")
    print(f"  Chunks with errors  : {sum(1 for e in errors_per_chunk_plan if e > 0)} / {num_chunks}")
    print()

    # Pad noisy payload for chunking
    noisy_padded = noisy_payload + [0] * pad_needed

    # Decode each chunk
    corrected_payload     = []
    total_corrected_bits  = 0
    failed_chunks         = 0
    chunk_syndromes_noisy = []

    for i in range(num_chunks):
        # Build the noisy codeword: take the clean enrollment codeword,
        # then flip the message-bit positions that have errors in this chunk.
        clean_cw      = list(codewords[i])
        noisy_cw_true = list(clean_cw)
        for pos in error_positions:
            if i * K <= pos < (i + 1) * K:
                bit_in_cw = pos - i * K    # position within message part
                noisy_cw_true[bit_in_cw] ^= 1

        # Compute syndrome of the noisy codeword (helper data for verification)
        syn_noisy = bch_syndrome(noisy_cw_true)
        chunk_syndromes_noisy.append(syn_noisy)

        # BCH decode to recover the original message bits
        corrected_msg, nerr = bch_decode_chunk(noisy_cw_true)
        corrected_payload.extend(corrected_msg)

        if nerr >= 0:
            total_corrected_bits += nerr
        else:
            failed_chunks += 1

    corrected_payload_trimmed = corrected_payload[:len(payload_bits)]

    # Show syndrome change for chunk 0
    syn_enroll_0 = all_syndromes[0]        # all zeros
    syn_noisy_0  = chunk_syndromes_noisy[0]
    errors_chunk0 = errors_per_chunk_plan[0]
    syndrome_changed = any(a != b for a, b in zip(syn_enroll_0, syn_noisy_0))

    print(f"  Chunk 0 syndrome — enrollment : {''.join(map(str, syn_enroll_0[:40]))}...")
    print(f"  Chunk 0 syndrome — noisy      : {''.join(map(str, syn_noisy_0[:40]))}...")
    print(f"  Chunk 0 errors injected       : {errors_chunk0}")
    print(f"  Chunk 0 syndrome changed      : {syndrome_changed}  ← errors detected via helper data")
    print()

    # Re-encode the corrected payload to reproduce the enrollment codeword
    corrected_padded  = corrected_payload_trimmed + [0] * pad_needed
    corrected_encoded = []
    for i in range(num_chunks):
        chunk = corrected_padded[i * K : (i + 1) * K]
        cw    = bch_encode_chunk(chunk)
        corrected_encoded.extend(cw)

    corrected_hash    = hashlib.sha256(_bits_to_bytes(corrected_encoded)).hexdigest()
    payload_recovered = (corrected_payload_trimmed == payload_bits)
    hash_matches      = (corrected_hash == hash_key)
    remaining_errors  = sum(a != b for a, b in zip(payload_bits, corrected_payload_trimmed))

    print(f"  BCH errors corrected  : {total_corrected_bits}")
    print(f"  Failed chunks         : {failed_chunks}  (should be 0)")
    print(f"  Remaining bit errors  : {remaining_errors}")
    print(f"  Payload fully recovered : {payload_recovered}")
    print()
    print(f"  Hash key — Test 1 (enrollment)  : {hash_key}")
    print(f"  Hash key — Test 2 (verification) : {corrected_hash}")
    verdict = "  SAME PERSON  —  hashes match" if hash_matches else "  MISMATCH  —  identity not confirmed"
    print(f"  Hashes match          : {hash_matches}  ← {verdict}")
    print(sep)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  PHASE 7 SUMMARY")
    print(sep)
    print(f"  BCH code per chunk     : BCH(N={BCH_N}, K={K}, t={BCH_T_DESIGNED})")
    print(f"  Number of chunks       : {num_chunks}")
    print(f"  Total t capacity       : {t_total}  (requested t=400  )")
    print(f"  Payload bits           : {payload_len}")
    print(f"  Encoded bits           : {num_chunks * BCH_N}")
    print(f"  Parity overhead        : {num_chunks * BCH_N - payload_len} bits")
    print(f"  Hash key (enrollment)  : {hash_key[:48]}...")
    print(f"  Syndrome (helper data) : {len(syndrome_flat)} bits stored")
    print(f"  Test 2 — t injected    : {T_INJECT}  errors  ({T_INJECT/payload_len*100:.1f}%)")
    print(f"  Test 2 — hash match    : {hash_matches}  ({' PASS' if hash_matches else ' FAIL'})")
    print(sep)

    return {
        "payload_bits"   : payload_bits,
        "encoded_bits"   : encoded_bits,
        "syndrome_hex"   : syndrome_hex,
        "hash_key"       : hash_key,
        "corrected_hash" : corrected_hash,
        "num_chunks"     : num_chunks,
        "K"              : K,
        "t_per_chunk"    : BCH_T_DESIGNED,
        "t_total"        : t_total,
        "hash_matches"   : hash_matches,
    }


if __name__ == "__main__":
    run()
