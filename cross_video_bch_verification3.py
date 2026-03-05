"""
ADAFACE FACE EMBEDDING PIPELINE + BCH FUZZY COMMITMENT
Phase 14 - CORRECTED TWO-FACTOR AUTHENTICATION
"""

import hashlib
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


# ==============================================================================
#  CONFIG - all tuneable parameters in one place, nothing hardcoded
# ==============================================================================

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",
]

WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)
OUTPUT_ROOT = "masked_frames"

FRAMES_TO_USE        = 20
CANDIDATE_MULTIPLIER = 3
FACE_SIZE            = 112
MASK_FRACTION        = 0.38

SIMILARITY_GATE_THRESHOLD = 0.75

BCH_N          = 255
BCH_T_DESIGNED = 35
QUANT_BITS     = 5


# ==============================================================================
#  SECTION 1 - FACE PIPELINE
# ==============================================================================

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
        img  = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img  = img.transpose(2, 0, 1)[np.newaxis]
        out  = self.session.run([self.output_name], {self.input_name: img})
        emb  = out[0][0] if out[0].ndim == 2 else out[0]
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            raise ValueError("Near-zero embedding norm.")
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
    log.info(f"  {Path(video_path).name} - {total} frames @ {fps:.1f} fps")
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
    print(f"\n  FINAL EMBEDDING - {video_name}  (512 dimensions)")
    print("  " + "-" * 60)
    for i in range(512):
        v = embedding[i]
        print(f"  Dim {i+1:3d}: {'+' if v >= 0 else ''}{v:.8f}")
    print("  " + "-" * 60)
    print(f"  Embedding norm: {np.linalg.norm(embedding):.8f}")
    print("  " + "-" * 60)


def process_video(
    video_path  : str,
    video_index : int,
    model       : AdaFaceModel,
    detector    : FaceDetector,
) -> Optional[Tuple[str, np.ndarray]]:
    name       = Path(video_path).name
    video_name = f"video_{video_index}"
    sep        = "-" * 60
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
            log.warning(f"  Frame {pos:>5}: no face - skipped")

    if not crops:
        log.error(f"  No faces found in {name}")
        return None
    log.info(f"  Valid face crops: {len(crops)}/{len(frames)}")

    emb_list    = []
    best_area   = 0
    best_masked = None
    for pos, crop in crops:
        resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
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
    log.info(f"  Masked photo saved -> {save_path}")

    stack = np.stack(emb_list, axis=0)
    avg   = np.mean(stack, axis=0).astype(np.float32)
    norm  = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Averaged embedding norm near zero.")
    final = (avg / norm).astype(np.float32)

    vis = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)
    print(f"\n  Frames extracted  : {len(frames)}")
    print(f"  Faces detected    : {len(crops)}/{len(frames)}")
    print(f"  Mask cut row      : {vis} of {FACE_SIZE}")
    print(f"  Saved photo       : {save_path}")
    print_embedding(final, video_name)
    return video_name, final


# ==============================================================================
#  SECTION 2 - COSINE SIMILARITY
# ==============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray, label: str = "") -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if abs(na - 1.0) > 1e-4: a = a / na
    if abs(nb - 1.0) > 1e-4: b = b / nb
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if label:
        log.info(f"Cosine similarity [{label}] : {sim:.8f}")
    return sim


def compute_pairwise_similarities(
    emb_dict  : Dict[str, np.ndarray],
    labels    : Dict[str, str],
    threshold : float,
) -> Dict[str, float]:
    keys = list(emb_dict.keys())
    n    = len(keys)
    sims = {}

    sep_eq = "=" * 70
    sep_dh = "-" * 70

    print(f"\n{sep_eq}")
    print("  PAIRWISE COSINE SIMILARITY COMPARISONS")
    print(f"  ({n*(n-1)//2} pairs across {n} videos  |  gate threshold = {threshold})")
    print(sep_eq)

    for v1n, v2n in itertools.combinations(keys, 2):
        e1, e2  = emb_dict[v1n], emb_dict[v2n]
        label1  = labels.get(v1n, v1n)
        label2  = labels.get(v2n, v2n)
        sim     = cosine_similarity(e1, e2, label=f"{v1n}_vs_{v2n}")
        key     = f"{v1n}_vs_{v2n}"
        verdict = "HIGH" if sim >= threshold else "LOW"
        sims[key] = sim
        print(f"  - {v1n} [{label1}] <-> {v2n} [{label2}]: {sim:.8f}  {verdict}")
        if sim < threshold:
            print(f"    Possible reasons for low similarity:")
            print(f"    - Different person in some videos")
            print(f"    - Poor quality frames in one video")
            print(f"    - Extreme pose/lighting variations")
            print(f"    - Face detection/alignment issues")
        print(sep_dh)

    # Per-video summary sections
    for focus in keys:
        focus_label = labels.get(focus, focus)
        vid_num     = focus.replace("video_", "")
        print()
        print(sep_eq)
        print(f"  VIDEO {vid_num} ({focus_label}) -- COMPARISONS SUMMARY")
        print(sep_eq)
        for other in keys:
            if other == focus:
                continue
            other_label = labels.get(other, other)
            key_fwd = f"{focus}_vs_{other}"
            key_rev = f"{other}_vs_{focus}"
            sim = sims.get(key_fwd) or sims.get(key_rev)
            if sim is None:
                continue
            verdict = "HIGH" if sim >= threshold else "LOW"
            print(
                f"  {focus} <-> {other}  [{other_label}]"
                f"  :  {sim:.8f}  {verdict}"
            )

    return sims


# ==============================================================================
#  SECTION 3 - GF(2) / GF(2^8) ARITHMETIC
# ==============================================================================

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
        if b & 1: result ^= a
        a <<= 1
        if a & 0x100: a ^= prim
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
        seen.append(e); e = (e * 2) % 255
    return seen


def _minimal_poly(root_exp: int) -> list:
    alpha = 2; conj = _conjugacy_class(root_exp); poly = [1]
    for e in conj:
        rv   = _gf256_pow(alpha, e)
        new  = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new[i]   ^= c
            new[i+1] ^= _gf256_mul(c, rv)
        poly = new
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


# ==============================================================================
#  SECTION 4 - BCH ENCODE / DECODE
# ==============================================================================

def bch_encode(msg_bits: list, g: list, K: int, PAR: int) -> list:
    assert len(msg_bits) == K
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
    assert len(received_bits) == BCH_N

    GF_EXP = [0] * 512; GF_LOG = [0] * 256; x = 1
    for i in range(255):
        GF_EXP[i] = GF_EXP[i + 255] = x; GF_LOG[x] = i; x = _gf256_mul(x, 2)

    def gmul(a, b): return 0 if (a == 0 or b == 0) else GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 255]
    def ginv(a):    return GF_EXP[255 - GF_LOG[a]]

    syndromes = []
    for i in range(1, 2 * t + 1):
        ai = GF_EXP[i % 255]; s = 0
        for bit in received_bits:
            s = gmul(s, ai) ^ bit
        syndromes.append(s)

    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0

    C = [1]+[0]*(2*t); B = [1]+[0]*(2*t); L = 0; m = 1; b = 1
    for n in range(2 * t):
        d = syndromes[n]
        for j in range(1, L + 1):
            if C[j] and syndromes[n-j]: d ^= gmul(C[j], syndromes[n-j])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = list(C); coef = gmul(d, ginv(b))
            for j in range(m, 2*t+1):
                if j-m < len(B) and B[j-m]: C[j] ^= gmul(coef, B[j-m])
            L = n+1-L; B = T; b = d; m = 1
        else:
            coef = gmul(d, ginv(b))
            for j in range(m, 2*t+1):
                if j-m < len(B) and B[j-m]: C[j] ^= gmul(coef, B[j-m])
            m += 1

    Lambda = C[:L+1]
    if L > t or L == 0:
        return list(received_bits[:K]), -1

    error_positions = []
    for j in range(1, BCH_N + 1):
        val = Lambda[0]
        for k in range(1, len(Lambda)):
            if Lambda[k]: val ^= gmul(Lambda[k], GF_EXP[(j*k) % 255])
        if val == 0:
            p = j - 1
            if 0 <= p < BCH_N: error_positions.append(p)

    if len(error_positions) != L:
        return list(received_bits[:K]), -1

    corrected = list(received_bits)
    for p in error_positions:
        corrected[p] ^= 1
    return corrected[:K], len(error_positions)


# ==============================================================================
#  SECTION 5 - QUANTISATION
# ==============================================================================

def embedding_to_payload(
    emb       : np.ndarray,
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


# ==============================================================================
#  SECTION 6 - INTERLEAVING
# ==============================================================================

def interleave_bits(payload_bits: list, num_chunks: int, K: int) -> List[list]:
    total = num_chunks * K
    assert len(payload_bits) == total, f"Expected {total} bits, got {len(payload_bits)}"
    chunks = [[] for _ in range(num_chunks)]
    for j, bit in enumerate(payload_bits):
        chunks[j % num_chunks].append(bit)
    for i, ch in enumerate(chunks):
        assert len(ch) == K, f"Chunk {i}: {len(ch)} bits, expected {K}"
    return chunks


def deinterleave_bits(chunks: List[list], num_chunks: int, K: int) -> list:
    result = [0] * (num_chunks * K)
    for ci, chunk in enumerate(chunks):
        for pos, bit in enumerate(chunk):
            result[pos * num_chunks + ci] = bit
    return result


# ==============================================================================
#  SECTION 7 - BIT HELPERS
# ==============================================================================

def bytes_to_bits(data: bytes, n_bits: int) -> list:
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    bits = bits[:n_bits]
    while len(bits) < n_bits:
        bits.append(0)
    return bits


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


# ==============================================================================
#  SECTION 8 - CORRECTED ENROLL + TWO-FACTOR VERIFY
# ==============================================================================

def bch_enroll(
    v1_embedding    : np.ndarray,
    v1_payload_bits : list,
    g               : list,
    K               : int,
    PAR             : int,
    num_chunks      : int,
    payload_bits_len: int,
) -> Tuple[List[list], str, np.ndarray]:
    pad_needed = (num_chunks * K) - len(v1_payload_bits)
    padded     = list(v1_payload_bits) + [0] * pad_needed
    p_chunks   = interleave_bits(padded, num_chunks, K)

    n_secret_bytes = math.ceil(num_chunks * K / 8)
    s_raw          = os.urandom(n_secret_bytes)
    s_bits         = bytes_to_bits(s_raw, num_chunks * K)
    s_chunks       = [s_bits[i * K : (i + 1) * K] for i in range(num_chunks)]

    helper_data = []
    for i in range(num_chunks):
        c_s      = bch_encode(s_chunks[i], g, K, PAR)
        p_pad255 = p_chunks[i] + [0] * PAR
        helper   = [a ^ b for a, b in zip(c_s, p_pad255)]
        helper_data.append(helper)

    hash_s = hashlib.sha256(bits_to_bytes(s_bits[:payload_bits_len])).hexdigest()

    ok = all(
        all(x == 0 for x in _gf2_divmod(bch_encode(s_chunks[i], g, K, PAR), g))
        for i in range(num_chunks)
    )
    log.info(f"Enrollment - all S codeword syndromes zero: {ok}  <- must be True")
    log.info(f"Enrollment - random secret S generated ({n_secret_bytes} bytes)")

    return helper_data, hash_s, v1_embedding.copy()


def two_factor_verify(
    probe_embedding  : np.ndarray,
    enrolled_template: np.ndarray,
    probe_payload    : list,
    helper_data      : List[list],
    hash_s_enroll    : str,
    g                : list,
    K                : int,
    PAR              : int,
    t                : int,
    num_chunks       : int,
    payload_bits_len : int,
    video_label      : str,
    gate_threshold   : float,
) -> dict:
    sep = "-" * 62

    sim         = cosine_similarity(probe_embedding, enrolled_template)
    gate_pass   = sim >= gate_threshold
    gate_margin = sim - gate_threshold

    print(f"  Factor 1 - Cosine Similarity Gate")
    print(f"  Probe vs enrolled template : {sim:.8f}")
    print(f"  Gate threshold             : {gate_threshold:.4f}")
    print(f"  Margin                     : {gate_margin:+.8f}  "
          f"({'ABOVE' if gate_pass else 'BELOW'} threshold)")
    gate_str = "PASS -> proceed to BCH" if gate_pass else "FAIL -> REJECT (impostor)"
    print(f"  Factor 1 decision          : {gate_str}")
    print(sep)

    if not gate_pass:
        print(f"  REJECTED by cosine gate - Factor 1 FAILED")
        print(f"  BCH not attempted. Impostor correctly identified.")
        print(sep)
        return {
            "label"        : video_label,
            "similarity"   : sim,
            "gate_pass"    : False,
            "bch_attempted": False,
            "failed_chunks": None,
            "corrected"    : None,
            "bch_pass"     : False,
            "overall_pass" : False,
        }

    print(f"  Factor 2 - BCH Fuzzy Commitment")

    pad_probe = (num_chunks * K) - len(probe_payload)
    q_padded  = list(probe_payload) + [0] * pad_probe
    q_chunks  = interleave_bits(q_padded, num_chunks, K)

    recovered_s_chunks = []
    total_corr         = 0
    failed_chunks_list = []

    for i in range(num_chunks):
        q_chunk  = q_chunks[i]
        q_pad255 = q_chunk + [0] * PAR
        noisy    = [a ^ b for a, b in zip(helper_data[i], q_pad255)]
        s_hat, nerr = bch_decode(noisy, g, K, PAR, t)
        if nerr >= 0:
            total_corr += nerr
            recovered_s_chunks.append(s_hat)
        else:
            failed_chunks_list.append(i)
            recovered_s_chunks.append([0] * K)

    n_failed         = len(failed_chunks_list)
    recovered_s_flat = []
    for ch in recovered_s_chunks:
        recovered_s_flat.extend(ch)

    hash_verify = hashlib.sha256(
        bits_to_bytes(recovered_s_flat[:payload_bits_len])
    ).hexdigest()
    bch_pass = (hash_verify == hash_s_enroll) and (n_failed == 0)
    overall  = gate_pass and bch_pass

    print(f"  BCH errors corrected total : {total_corr}")
    print(f"  Failed chunks              : {n_failed} / {num_chunks}"
          f"  (0 = full secret recovery)")
    if failed_chunks_list:
        print(f"  Failed chunk indices       : {str(failed_chunks_list[:10])}")
    print(f"  Hash enrolled (S)          : {hash_s_enroll[:32]}...")
    print(f"  Hash recovered (S_hat)     : {hash_verify[:32]}...")
    bch_str = "PASS  hashes match" if bch_pass else "FAIL  hashes differ"
    print(f"  Factor 2 decision          : {bch_str}")
    print(sep)

    if overall:
        print(f"  ACCEPTED - Factor 1 PASSED + Factor 2 PASSED")
        print(f"  Identity confirmed. Secret key recovered.")
    else:
        reasons = []
        if not gate_pass: reasons.append("Factor 1 FAILED (cosine gate)")
        if not bch_pass:  reasons.append("Factor 2 FAILED (BCH hash mismatch)")
        print(f"  REJECTED - {', '.join(reasons)}")
    print(sep)

    return {
        "label"        : video_label,
        "similarity"   : sim,
        "gate_pass"    : gate_pass,
        "bch_attempted": True,
        "failed_chunks": n_failed,
        "corrected"    : total_corr,
        "bch_pass"     : bch_pass,
        "overall_pass" : overall,
    }


# ==============================================================================
#  SECTION 9 - CROSS-ENROLLMENT TEST (V5 vs V4 helper data)
# ==============================================================================

def cross_enrollment_test(
    embeddings    : Dict[str, np.ndarray],
    g             : list,
    BCH_K         : int,
    BCH_PAR       : int,
    NUM_CHUNKS    : int,
    PAYLOAD_BITS  : int,
    gate_threshold: float,
) -> None:
    sep_eq = "=" * 70
    sep_dh = "-" * 70

    print(f"\n{sep_eq}")
    print("  CROSS-ENROLLMENT TEST: V5 (IMPOSTOR) vs V4 HELPER DATA")
    print("  Enrolling V4 as reference. Probing with V5.")
    print("  Expected outcome: REJECT at Factor 1 (cosine gate).")
    print(sep_eq)

    if "video_4" not in embeddings or "video_5" not in embeddings:
        print("  ERROR: video_4 or video_5 not available - skipping cross test.")
        return

    v4_bits, _, v4_min, v4_max = embedding_to_payload(embeddings["video_4"])
    log.info(f"Cross-test: V4 quantised - scale [{v4_min:.5f}, {v4_max:.5f}]")

    print(f"\n{sep_dh}")
    print("  CROSS-ENROLLMENT - V4 (Android No Beard) as reference")
    print(sep_dh)

    helper_v4, hash_s_v4, template_v4 = bch_enroll(
        v1_embedding     = embeddings["video_4"],
        v1_payload_bits  = v4_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks       = NUM_CHUNKS,
        payload_bits_len = PAYLOAD_BITS,
    )

    helper_hex = bits_to_hex([b for hd in helper_v4 for b in hd])
    print(f"  Enrolled identity        : V4 - Android No Beard")
    print(f"  Hash of S (SHA-256)      : {hash_s_v4}")
    print(f"  Helper (first 64 hex)    : {helper_hex[:64]}...")
    print(f"  V4 scale                 : [{v4_min:.5f}, {v4_max:.5f}]")
    print(sep_dh)

    print(f"\n{sep_dh}")
    print("  CROSS-VERIFICATION - V5 (impostor) probing V4 helper data")
    print(sep_dh)

    probe_bits_v5, _, _, _ = embedding_to_payload(
        embeddings["video_5"], shared_min=v4_min, shared_max=v4_max
    )
    log.info(
        f"Cross-test: V5 quantised using V4 scale [{v4_min:.5f}, {v4_max:.5f}]"
    )

    result = two_factor_verify(
        probe_embedding  = embeddings["video_5"],
        enrolled_template= template_v4,
        probe_payload    = probe_bits_v5,
        helper_data      = helper_v4,
        hash_s_enroll    = hash_s_v4,
        g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
        num_chunks       = NUM_CHUNKS,
        payload_bits_len = PAYLOAD_BITS,
        video_label      = "video_5 vs V4 helper",
        gate_threshold   = gate_threshold,
    )

    print(f"\n{sep_eq}")
    print("  CROSS-ENROLLMENT TEST RESULT")
    print(sep_eq)
    print(f"  Probe              : V5 (Android video 5 - DIFFERENT PERSON)")
    print(f"  Reference enrolled : V4 (Android No Beard - genuine user)")
    print(f"  Cosine similarity  : {result['similarity']:.8f}")
    print(f"  Factor 1 gate      : {'PASS' if result['gate_pass'] else 'FAIL'}")
    print(f"  BCH attempted      : {'YES' if result['bch_attempted'] else 'NO'}")
    print(f"  Final decision     : {'ACCEPTED' if result['overall_pass'] else 'REJECTED'}")
    if not result["overall_pass"]:
        print(f"  Security verdict   : CORRECT - impostor rejected when probing V4 identity")
    else:
        print(f"  Security verdict   : BREACH - impostor accepted against V4 identity")
    print(sep_eq)


# ==============================================================================
#  SECTION 10 - MAIN
# ==============================================================================

def run():
    sep = "=" * 70

    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial ...")
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    PAYLOAD_BITS = 512 * QUANT_BITS
    NUM_CHUNKS   = math.ceil(PAYLOAD_BITS / BCH_K)
    T_TOTAL      = NUM_CHUNKS * BCH_T_DESIGNED
    PAD_NEEDED   = NUM_CHUNKS * BCH_K - PAYLOAD_BITS
    RATE_PCT     = BCH_K / BCH_N * 100

    print(sep)
    print("  ADAFACE + BCH FUZZY-COMMITMENT - Phase 14  (Two-Factor Auth)")
    print(sep)
    print(f"  Videos processed : {len(VIDEO_PATHS)}")
    for i, vp in enumerate(VIDEO_PATHS, 1):
        print(f"    {i}. {Path(vp).name}")
    print()
    print("  SECURITY FIXES vs PHASE 13")
    print("  FIX 1 - Random secret S (not biometric) bound in BCH codeword")
    print("    Phase 13 flaw: secret r[i] = V1_chunk[i]  (biometric itself)")
    print("    Fix: S = os.urandom(...), S has NO relation to the face")
    print("    Effect: BCH recovering V1 bits from impostor no longer")
    print("    produces the enrolled secret -> hash mismatch -> FAIL")
    print()
    print("  FIX 2 - Cosine similarity gate as primary identity check")
    print("    Phase 13 flaw: BCH alone cannot reject impostors because")
    print("    genuine max_chunk=32 > impostor avg_chunk=21.9 -> no valid t")
    print("    Fix: gate rejects any probe with cosine < threshold BEFORE BCH")
    print(f"    Threshold = {SIMILARITY_GATE_THRESHOLD}  "
          f"(genuine min=0.811, impostor max=0.201, gap=0.610)")
    print()
    print("  BCH PARAMETERS")
    print(f"    BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})  "
          f"PAR={BCH_PAR}  derived from t - NOT hardcoded")
    print(f"    Rate={RATE_PCT:.1f}%  Chunks={NUM_CHUNKS}  "
          f"t_total={T_TOTAL}  Padding={PAD_NEEDED} bits")
    print(f"    QUANT_BITS={QUANT_BITS}  Payload={PAYLOAD_BITS} bits  "
          f"Chunking=INTERLEAVED")
    print(sep)

    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    VIDEO_LABELS = {
        "video_1": "IOS",
        "video_2": "IOS_NoBeard",
        "video_3": "Android",
        "video_4": "Android_NoBeard",
        "video_5": "Android_v5",
    }

    embeddings: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        result = process_video(vp, idx, model, detector)
        if result:
            name, emb = result
            embeddings[name] = emb

    print(f"\n{sep}")
    print("  PROCESSING COMPLETE - FINAL EMBEDDING NORMS")
    print(sep)
    for name, emb in embeddings.items():
        print(f"  {name:10}: norm = {np.linalg.norm(emb):.8f}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep)

    similarities: Dict[str, float] = {}
    if len(embeddings) >= 2:
        similarities = compute_pairwise_similarities(
            embeddings,
            labels    = VIDEO_LABELS,
            threshold = SIMILARITY_GATE_THRESHOLD,
        )
        if similarities:
            sv    = list(similarities.values())
            above = sum(1 for s in sv if s >= SIMILARITY_GATE_THRESHOLD)
            below = len(sv) - above
            print()
            print(f"  AVERAGE SIMILARITY (all {len(sv)} pairs) : {np.mean(sv):.8f}")
            quality = "GOOD" if below == 0 else f"POOR ({below} pairs below threshold)"
            print(f"  OVERALL QUALITY: {quality}")
            print(f"  Pairs above gate ({SIMILARITY_GATE_THRESHOLD}): {above}")
            print(f"  Pairs below gate ({SIMILARITY_GATE_THRESHOLD}): {below}"
                  f"  <- expected to be different-person comparisons")

    if "video_1" not in embeddings:
        print(f"\n{sep}\n  ERROR: video_1 not available - cannot enroll.\n{sep}")
        return embeddings, similarities

    print(f"\n{'=' * 70}")
    print("  PHASE 14 - TWO-FACTOR BCH FUZZY-COMMITMENT")
    print("  ENROLL: V1  |  VERIFY: V2, V3, V4 (genuine) + V5 (impostor)")
    print(f"{'=' * 70}")

    v1_bits, _, v1_min, v1_max = embedding_to_payload(embeddings["video_1"])
    log.info(
        f"V1 quantised - scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
        f"payload = {len(v1_bits)} bits"
    )

    print(f"\n{'-' * 62}")
    print("  ENROLLMENT - V1 (IOS Beard)")
    print(f"{'-' * 62}")

    helper_data, hash_s_H1, enrolled_template = bch_enroll(
        v1_embedding     = embeddings["video_1"],
        v1_payload_bits  = v1_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks       = NUM_CHUNKS,
        payload_bits_len = PAYLOAD_BITS,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Payload bits         : {len(v1_bits)}")
    print(f"  Secret S             : {math.ceil(NUM_CHUNKS*BCH_K/8)} random bytes"
          f"  <- DISCARDED after enrollment, never stored")
    print(f"  Enrolled template    : stored (512-dim unit vector)"
          f"  <- Factor 1 gate reference")
    print(f"  Chunks               : {NUM_CHUNKS}  (interleaved, K={BCH_K})")
    print(f"  Helper data size     : {NUM_CHUNKS} x {BCH_N} = {NUM_CHUNKS*BCH_N} bits")
    print(f"  Hash of S (SHA-256)  : {hash_s_H1}")
    print(f"  Helper (first 64 hex): {helper_hex[:64]}...")
    print(f"  V1 scale             : [{v1_min:.5f}, {v1_max:.5f}]")
    print(f"  Stored data          : helper_data, hash_S, template, v_min, v_max")
    print(f"  NOT stored           : raw secret S, raw embedding (beyond template)")
    print(f"{'-' * 62}")

    video_labels_verify = {
        "video_2": "IOS No Beard     (V2) - same person",
        "video_3": "Android Beard    (V3) - same person",
        "video_4": "Android No Beard (V4) - same person",
        "video_5": "Android Video 5  (V5) - DIFFERENT PERSON",
    }
    summary = []

    for vid in ["video_2", "video_3", "video_4", "video_5"]:
        if vid not in embeddings:
            log.error(f"{vid} not available - skipping.")
            continue

        label = video_labels_verify[vid]
        print(f"\n{'-' * 62}")
        print(f"  VERIFICATION - {label}")
        print(f"{'-' * 62}")

        probe_bits, _, _, _ = embedding_to_payload(
            embeddings[vid], shared_min=v1_min, shared_max=v1_max
        )
        log.info(
            f"{vid} quantised - scale [{v1_min:.5f}, {v1_max:.5f}]  |  "
            f"payload = {len(probe_bits)} bits"
        )

        result = two_factor_verify(
            probe_embedding  = embeddings[vid],
            enrolled_template= enrolled_template,
            probe_payload    = probe_bits,
            helper_data      = helper_data,
            hash_s_enroll    = hash_s_H1,
            g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
            num_chunks       = NUM_CHUNKS,
            payload_bits_len = PAYLOAD_BITS,
            video_label      = vid,
            gate_threshold   = SIMILARITY_GATE_THRESHOLD,
        )
        summary.append(result)

    # Cross-enrollment: V5 attempting to use V4 enrolled helper data
    cross_enrollment_test(
        embeddings    = embeddings,
        g             = g,
        BCH_K         = BCH_K,
        BCH_PAR       = BCH_PAR,
        NUM_CHUNKS    = NUM_CHUNKS,
        PAYLOAD_BITS  = PAYLOAD_BITS,
        gate_threshold = SIMILARITY_GATE_THRESHOLD,
    )

    print(f"\n{'=' * 70}")
    print("  PHASE 14 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Enrollment   : V1 - IOS Beard")
    print(f"  Hash of S    : {hash_s_H1}")
    print(f"  Secret S     : discarded at enrollment - never stored")
    print(f"  BCH params   : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})"
          f" x {NUM_CHUNKS} chunks  t_total={T_TOTAL}")
    print(f"  Rate={RATE_PCT:.1f}%  PAR={BCH_PAR}  Chunking=INTERLEAVED  "
          f"QUANT_BITS={QUANT_BITS}")
    print(f"  Gate threshold         : {SIMILARITY_GATE_THRESHOLD}")
    print(f"  V1 scale               : [{v1_min:.5f}, {v1_max:.5f}]")
    print()

    col = (
        f"  {'Video':<44}  {'Cosine':>8}  {'F1 Gate':>7}"
        f"  {'BCH':>4}  {'F2 BCH':>6}  {'DECISION':>10}"
    )
    print(col)
    print(
        f"  {'-'*44}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*6}  {'-'*10}"
    )
    for r in summary:
        f1  = "PASS" if r["gate_pass"]     else "FAIL"
        bch = "YES"  if r["bch_attempted"] else "NO"
        f2  = ("PASS" if r["bch_pass"] else "FAIL") if r["bch_attempted"] else "SKIP"
        dec = "ACCEPT" if r["overall_pass"] else "REJECT"
        print(
            f"  {r['label']:<44}  {r['similarity']:>8.6f}  {f1:>7}"
            f"  {bch:>4}  {f2:>6}  {dec:>10}"
        )

    genuine_results   = [r for r in summary if "DIFFERENT" not in r["label"]]
    impostor_results  = [r for r in summary if "DIFFERENT"     in r["label"]]
    genuine_accepted  = sum(1 for r in genuine_results  if r["overall_pass"])
    impostor_rejected = sum(1 for r in impostor_results if not r["overall_pass"])
    genuine_total     = len(genuine_results)
    impostor_total    = len(impostor_results)

    print()
    print(f"  Genuine  users ACCEPTED : {genuine_accepted} / {genuine_total}")
    print(f"  Impostor users REJECTED : {impostor_rejected} / {impostor_total}")
    print()

    if genuine_accepted == genuine_total and impostor_rejected == impostor_total:
        print("  SYSTEM SECURE")
        print("  All genuine users accepted.  All impostors rejected.")
        print("  Cosine gate stopped impostor before BCH was attempted.")
        print("  Random secret S means BCH result is cryptographically")
        print("  meaningful - not just reconstruction of the biometric.")
    elif impostor_rejected < impostor_total:
        print("  SECURITY BREACH - impostor not rejected")
        print("  Raise SIMILARITY_GATE_THRESHOLD or lower BCH_T_DESIGNED.")
    else:
        print("  Partial result - some genuine users rejected.")
        print("  Check SIMILARITY_GATE_THRESHOLD (may be too high) or")
        print("  BCH_T_DESIGNED (may be too low for your embedding noise).")

    print(f"{'=' * 70}")
    return embeddings, similarities


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY-COMMITMENT - Phase 14  (Two-Factor Auth)")
    print("  Security fixes: random secret S + cosine gate")
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
        print(f"  Similarity range  : {min(sv):.8f} - {max(sv):.8f}")
        low = [s for s in sv if s < SIMILARITY_GATE_THRESHOLD]
        print(f"  Pairs below gate  : {len(low)} / {len(sv)}"
              f"  <- expected to be different-person comparisons")

    print("=" * 70)
