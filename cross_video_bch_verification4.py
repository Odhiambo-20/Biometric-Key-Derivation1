"""
ADAFACE + BCH FUZZY-COMMITMENT  -  Phase 10  (QUANT_BITS=5, t=35)

Videos:
  1. IOS.mov
  2. IOS M-No Beard .mov
  3. Android .mp4
  4. Android M-No Beard .mp4   <- ENROLLED  (V4)
  5. Android video 5.mp4       <- VERIFIED against V4  (V5)

Mask placement:
  - Covers ONLY: below nostrils -> mouth, chin  (bottom ~38%)
  - Keeps visible: forehead, eyebrows, eyes, nose INCLUDING NOSTRILS (top ~62%)

MASK_FRACTION = 0.38:
  black_rows = int(112 * 0.38) = 42 rows  -> rows 70..111 = black
  visible    = 112 - 42        = 70 rows  -> rows  0..69

High-quality frame selection:
  - Candidate pool: 60 evenly spaced frames
  - Quality metric: Laplacian variance (sharpness)
  - Final selection: top 20 sharpest frames

Authentication:
  Stage 1  Cosine gate  >= COSINE_GATE_THRESHOLD required to proceed
  Stage 2  BCH commit   SHA-256(BCH-recovered r) must match enrolled hash
  Both stages must pass for ACCEPT.

WHY THE COSINE GATE IS MANDATORY
  BCH(255, K=47, t=35) corrects up to t/K = 74.5% of message bits.
  A different-person probe has ~50% bit flip rate, which is below that
  ceiling. Without the gate, BCH silently forces any impostor back to the
  enrolled template and the hash always matches. The cosine gate is the
  true same-person / different-person discriminator.
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


# ===========================================================================
# CONFIG
# ===========================================================================

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",
]

VIDEO_LABELS = {
    "video_1": "IOS",
    "video_2": "IOS_NoBeard",
    "video_3": "Android",
    "video_4": "Android_NoBeard",
    "video_5": "Android_v5",
}

WEIGHTS_PATH             = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
OUTPUT_ROOT              = "masked_frames"
FRAMES_TO_USE            = 20
CANDIDATE_MULTIPLIER     = 3
FACE_SIZE                = 112
MASK_FRACTION            = 0.38
MIN_SIMILARITY_THRESHOLD = 0.80

# Probes with cosine below this value are rejected before BCH runs.
# Calibrated between genuine pairs (0.81-0.96) and impostors (0.10-0.20).
COSINE_GATE_THRESHOLD = 0.65

# BCH parameters
BCH_N          = 255
BCH_T_DESIGNED = 35
QUANT_BITS     = 4


# ===========================================================================
# SECTION 1 - GF(2^8) ARITHMETIC
# ===========================================================================

_PRIM_POLY = 0x11D   # x^8 + x^4 + x^3 + x^2 + 1


def _build_gf_tables():
    exp_table = [0] * 512
    log_table = [0] * 256
    x = 1
    for i in range(255):
        exp_table[i] = x
        log_table[x] = i
        x <<= 1
        if x >= 256:
            x ^= _PRIM_POLY
    for i in range(255, 512):
        exp_table[i] = exp_table[i - 255]
    return exp_table, log_table


_GF_EXP, _GF_LOG = _build_gf_tables()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] + _GF_LOG[b]) % 255]


def _gf_inv(a):
    if a == 0:
        raise ZeroDivisionError("GF inverse of 0 is undefined")
    return _GF_EXP[255 - _GF_LOG[a]]


# ===========================================================================
# SECTION 2 - BCH GENERATOR POLYNOMIAL
# ===========================================================================

def _cyclotomic_coset(i, n=255):
    coset = []
    seen  = set()
    x     = i % n
    while x not in seen:
        seen.add(x)
        coset.append(x)
        x = (2 * x) % n
    return coset


def _min_poly_gf2(coset):
    poly = [1]
    for j in coset:
        alpha_j  = _GF_EXP[j % 255] if j > 0 else 1
        new_poly = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            new_poly[k]     ^= _gf_mul(c, alpha_j)
            new_poly[k + 1] ^= c
        poly = new_poly
    return [c & 1 for c in poly]


def _poly_mul_gf2(a, b):
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                result[i + j] ^= bj
    return result


def build_bch_generator(t, n=255):
    covered = set()
    g = [1]
    for i in range(1, 2 * t + 1):
        if i % n in covered:
            continue
        coset = _cyclotomic_coset(i, n)
        covered.update(coset)
        g = _poly_mul_gf2(g, _min_poly_gf2(coset))
    while len(g) > 1 and g[-1] == 0:
        g.pop()
    PAR = len(g) - 1
    K   = n - PAR
    log.info(f"BCH(N={n}, K={K}, t={t}): deg(g) = {PAR}")
    return g, K, PAR


# ===========================================================================
# SECTION 3 - BCH ENCODE
# ===========================================================================

def bch_encode(message_bits, g, K, PAR):
    N = K + PAR
    assert len(message_bits) == K, f"Expected {K} bits, got {len(message_bits)}"
    rem = [0] * N
    for i, b in enumerate(message_bits):
        rem[PAR + i] = b
    for i in range(N - 1, PAR - 1, -1):
        if rem[i] == 0:
            continue
        shift = i - PAR
        for j in range(PAR + 1):
            rem[shift + j] ^= g[j]
    return list(message_bits) + rem[:PAR]


# ===========================================================================
# SECTION 4 - BCH DECODE (Berlekamp-Massey + Chien search)
# ===========================================================================

def _compute_syndromes(received, t, n=255):
    syndromes = []
    for j in range(1, 2 * t + 1):
        s = 0
        for i, b in enumerate(received):
            if b:
                s ^= _GF_EXP[(i * j) % 255]
        syndromes.append(s)
    return syndromes


def _berlekamp_massey(syndromes, t):
    n = 2 * t
    C = [1] + [0] * n
    B = [1] + [0] * n
    L = 0
    m = 1
    b = 1
    for i in range(n):
        d = syndromes[i]
        for j in range(1, L + 1):
            if C[j] and syndromes[i - j]:
                d ^= _gf_mul(C[j], syndromes[i - j])
        if d == 0:
            m += 1
        elif 2 * L <= i:
            T     = C[:]
            coeff = _gf_mul(d, _gf_inv(b))
            for j in range(m, n + 1):
                if B[j - m]:
                    C[j] ^= _gf_mul(coeff, B[j - m])
            L = i + 1 - L
            B = T
            b = d
            m = 1
        else:
            coeff = _gf_mul(d, _gf_inv(b))
            for j in range(m, n + 1):
                if B[j - m]:
                    C[j] ^= _gf_mul(coeff, B[j - m])
            m += 1
    return C[:L + 1]


def _chien_search(sigma, n=255):
    positions = []
    for i in range(n):
        alpha_inv_i = _GF_EXP[(255 - i) % 255]
        val = 0
        xi  = 1
        for coeff in sigma:
            val ^= _gf_mul(coeff, xi)
            xi   = _gf_mul(xi, alpha_inv_i)
        if val == 0:
            positions.append(i)
    return positions


def bch_decode(received_bits, g, K, PAR, t):
    N = K + PAR
    assert len(received_bits) == N, f"Expected {N} bits, got {len(received_bits)}"
    syndromes = _compute_syndromes(received_bits, t)
    if all(s == 0 for s in syndromes):
        return list(received_bits[:K]), 0
    sigma    = _berlekamp_massey(syndromes, t)
    num_errs = len(sigma) - 1
    if num_errs > t:
        return list(received_bits[:K]), -1
    positions = _chien_search(sigma)
    if len(positions) != num_errs:
        return list(received_bits[:K]), -1
    corrected = list(received_bits)
    for pos in positions:
        if pos < N:
            corrected[pos] ^= 1
    if not all(s == 0 for s in _compute_syndromes(corrected, t)):
        return list(received_bits[:K]), -1
    return list(corrected[:K]), num_errs


# ===========================================================================
# SECTION 5 - BIT AND HEX HELPERS
# ===========================================================================

def bits_to_bytes(bits):
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


def bits_to_hex(bits):
    b = list(bits)
    while len(b) % 4 != 0:
        b.insert(0, 0)
    return "".join(
        format(b[i] * 8 + b[i + 1] * 4 + b[i + 2] * 2 + b[i + 3], "x")
        for i in range(0, len(b), 4)
    )


def _gf2_divmod(poly, divisor):
    rem   = list(poly)
    deg_d = len(divisor) - 1
    for i in range(len(rem) - 1, deg_d - 1, -1):
        if rem[i] == 0:
            continue
        shift = i - deg_d
        for j in range(len(divisor)):
            rem[shift + j] ^= divisor[j]
    result = rem[:deg_d]
    while len(result) < deg_d:
        result.append(0)
    return result


# ===========================================================================
# SECTION 6 - EMBEDDING QUANTIZATION
# ===========================================================================

def embedding_to_payload(embedding, shared_min=None, shared_max=None):
    v     = np.array(embedding, dtype=np.float32)
    v_min = float(v.min()) if shared_min is None else shared_min
    v_max = float(v.max()) if shared_max is None else shared_max
    levels  = 2 ** QUANT_BITS
    v_range = max(v_max - v_min, 1e-10)
    quantized = np.clip(
        np.floor((v - v_min) / v_range * levels).astype(int),
        0, levels - 1,
    )
    bits = []
    for q in quantized:
        for b in range(QUANT_BITS - 1, -1, -1):
            bits.append(int((q >> b) & 1))
    return bits, quantized.tolist(), v_min, v_max


# ===========================================================================
# SECTION 7 - FACE PIPELINE COMPONENTS
# ===========================================================================

def apply_mask(image):
    img        = image.copy()
    black_rows = int(image.shape[0] * MASK_FRACTION)
    img[-black_rows:, :] = 0
    return img


def sharpness_score(gray_frame):
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(xml)
        if self.det.empty():
            raise RuntimeError("Haar cascade XML not found.")
        log.info("Face detector ready.")

    def detect(self, frame):
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
    def __init__(self, model_path):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded | Provider: {providers[0]}")

    def get_embedding(self, face_112):
        img = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            raise ValueError("Embedding norm near zero - bad crop?")
        return (emb / norm).astype(np.float32)


def extract_high_quality_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    n_candidates = num_frames * CANDIDATE_MULTIPLIER
    log.info(f"  {Path(video_path).name} -- {total} frames @ {fps:.1f} fps")
    log.info(f"  Scanning {n_candidates} candidates, picking top {num_frames} by sharpness")
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates.append((sharpness_score(gray), pos, frame))
    cap.release()
    if not candidates:
        raise RuntimeError("No frames could be read from the video.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:num_frames]
    top.sort(key=lambda x: x[1])
    log.info(
        f"  Sharpness range (selected): {top[0][0]:.1f} .. {top[-1][0]:.1f}"
        f"  (pool max={candidates[0][0]:.1f})"
    )
    return [(pos, frame) for _, pos, frame in top]


def process_video(video_path, video_index, model, detector):
    sep        = "-" * 60
    name       = Path(video_path).name
    video_name = f"video_{video_index}"
    label      = VIDEO_LABELS.get(video_name, video_name)
    print(f"\n{sep}\n  VIDEO {video_index} [{label}]: {name}\n{sep}")

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
            log.warning(f"  Frame {pos:>5}: no face detected -- skipped")

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
        log.info(f"  Frame {pos:>5}: embedded | norm={np.linalg.norm(emb):.6f}")
        area = crop.shape[0] * crop.shape[1]
        if area > best_area:
            best_area   = area
            best_masked = masked

    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path  = output_dir / f"video_{video_index}_masked.jpg"
    cv2.imwrite(str(save_path), best_masked)
    log.info(f"  Masked photo saved -> {save_path}")

    stack = np.stack(embeddings, axis=0)
    avg   = np.mean(stack, axis=0).astype(np.float32)
    norm  = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Averaged embedding norm near zero.")
    final = (avg / norm).astype(np.float32)

    black_rows   = int(FACE_SIZE * MASK_FRACTION)
    visible_rows = FACE_SIZE - black_rows
    print(f"  Frames extracted : {len(frames)}")
    print(f"  Faces detected   : {len(crops)}")
    print(f"  Mask cut line    : row {visible_rows} of 112")
    print(f"  Saved photo      : {save_path}")

    return video_name, final


# ===========================================================================
# SECTION 8 - COSINE SIMILARITY
# ===========================================================================

def cosine_similarity(v1, v2, label=""):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if abs(n1 - 1.0) > 1e-4:
        log.warning(f"Vector 1 not unit vector (norm={n1:.6f}). Renormalizing.")
        v1 = v1 / n1
    if abs(n2 - 1.0) > 1e-4:
        log.warning(f"Vector 2 not unit vector (norm={n2:.6f}). Renormalizing.")
        v2 = v2 / n2
    sim = float(np.dot(v1, v2))
    sim = max(-1.0, min(1.0, sim))
    if label:
        log.info(f"Cosine similarity [{label}]: {sim:.8f}")
    return sim


def compute_pairwise_similarities(embeddings_dict):
    video_names  = list(embeddings_dict.keys())
    similarities = {}

    print("\n" + "=" * 60)
    print("  PAIRWISE COSINE SIMILARITY COMPARISONS")
    print("=" * 60)

    for v1_name, v2_name in itertools.combinations(video_names, 2):
        emb1   = embeddings_dict[v1_name]
        emb2   = embeddings_dict[v2_name]
        label1 = VIDEO_LABELS.get(v1_name, v1_name)
        label2 = VIDEO_LABELS.get(v2_name, v2_name)
        sim      = cosine_similarity(emb1, emb2, label=f"{v1_name}_vs_{v2_name}")
        pair_key = f"{v1_name}_vs_{v2_name}"
        similarities[pair_key] = sim
        status = "GOOD" if sim >= MIN_SIMILARITY_THRESHOLD else "LOW"
        print(f"  {v1_name} [{label1}]  vs  {v2_name} [{label2}]:  {sim:.8f}  {status}")

    return similarities


def analyze_similarities(similarities):
    if not similarities:
        return {}
    all_sims = list(similarities.values())
    max_pair = max(similarities.items(), key=lambda x: x[1])
    min_pair = min(similarities.items(), key=lambda x: x[1])
    low_sims = [(p, s) for p, s in similarities.items() if s < MIN_SIMILARITY_THRESHOLD]
    if len(low_sims) == 0:
        quality = "GOOD (all pairs above threshold)"
    elif len(low_sims) <= len(all_sims) / 3:
        quality = "FAIR (some pairs below threshold)"
    else:
        quality = "POOR (multiple pairs below threshold)"
    return {
        "all_similarities"       : all_sims,
        "min_similarity"         : min(all_sims),
        "max_similarity"         : max(all_sims),
        "mean_similarity"        : float(np.mean(all_sims)),
        "std_dev"                : float(np.std(all_sims)),
        "most_similar_pair"      : max_pair,
        "least_similar_pair"     : min_pair,
        "low_similarities_count" : len(low_sims),
        "low_similarities_pairs" : low_sims,
        "quality_assessment"     : quality,
        "total_pairs"            : len(all_sims),
    }


def print_similarity_analysis(analysis):
    if not analysis:
        return
    mp       = analysis["most_similar_pair"]
    lp       = analysis["least_similar_pair"]
    ml1, ml2 = mp[0].split("_vs_")
    ll1, ll2 = lp[0].split("_vs_")
    print("\n  SIMILARITY STATISTICS")
    print("-" * 60)
    print(f"  Min     : {analysis['min_similarity']:.8f}")
    print(f"  Max     : {analysis['max_similarity']:.8f}")
    print(f"  Mean    : {analysis['mean_similarity']:.8f}")
    print(f"  Std Dev : {analysis['std_dev']:.8f}")
    print(
        f"  Most similar  : {ml1} [{VIDEO_LABELS.get(ml1, ml1)}]"
        f" <-> {ml2} [{VIDEO_LABELS.get(ml2, ml2)}]  =  {mp[1]:.8f}"
    )
    print(
        f"  Least similar : {ll1} [{VIDEO_LABELS.get(ll1, ll1)}]"
        f" <-> {ll2} [{VIDEO_LABELS.get(ll2, ll2)}]  =  {lp[1]:.8f}"
    )
    if analysis["low_similarities_count"] > 0:
        print(
            f"\n  {analysis['low_similarities_count']}/{analysis['total_pairs']} "
            f"pairs below threshold {MIN_SIMILARITY_THRESHOLD}:"
        )
        for pair, sim in analysis["low_similarities_pairs"]:
            p1, p2 = pair.split("_vs_")
            print(
                f"     {p1} [{VIDEO_LABELS.get(p1, p1)}]"
                f" <-> {p2} [{VIDEO_LABELS.get(p2, p2)}]: {sim:.8f}"
            )
    print("-" * 60)


def print_v5_comparisons(similarities):
    v5_pairs = {k: v for k, v in similarities.items() if "video_5" in k}
    if not v5_pairs:
        return
    print("\n  VIDEO 5 (Android_v5) - COMPARISONS SUMMARY")
    print("-" * 60)
    for pair, sim in sorted(v5_pairs.items(), key=lambda x: x[1], reverse=True):
        p1, p2 = pair.split("_vs_")
        other  = p2 if p1 == "video_5" else p1
        lbl    = VIDEO_LABELS.get(other, other)
        status = "GOOD" if sim >= MIN_SIMILARITY_THRESHOLD else "LOW"
        print(f"  video_5 <-> {other:8} [{lbl:20}]:  {sim:.8f}  {status}")
    print("-" * 60)


# ===========================================================================
# SECTION 9 - BCH FUZZY-COMMITMENT ENROLL + VERIFY
# ===========================================================================

def bch_enroll(v_ref_payload_bits, g, K, PAR, num_chunks):
    """
    Enroll a reference video.

    Per chunk i:
        r[i]      = ref_chunk[i]
        c_r[i]    = BCH_encode(r[i])
        helper[i] = c_r[i] XOR (r[i] ++ zeros(PAR))

    hash_key = SHA-256(all r bits concatenated)
    """
    pad      = (num_chunks * K) - len(v_ref_payload_bits)
    v_padded = list(v_ref_payload_bits) + [0] * pad
    helper_data = []
    all_r_bits  = []
    for i in range(num_chunks):
        r       = v_padded[i * K : (i + 1) * K]
        c_r     = bch_encode(r, g, K, PAR)
        ref_pad = r + [0] * PAR
        helper_data.append([a ^ b for a, b in zip(c_r, ref_pad)])
        all_r_bits.extend(r)
    hash_key = hashlib.sha256(bits_to_bytes(all_r_bits)).hexdigest()
    ok = all(
        all(s == 0 for s in _gf2_divmod(bch_encode(v_padded[i*K:(i+1)*K], g, K, PAR), g))
        for i in range(num_chunks)
    )
    log.info(f"Enrollment - all codeword syndromes zero: {ok}  (must be True)")
    return helper_data, hash_key


def bch_verify(
    probe_embedding,
    enrolled_embedding,
    vx_payload_bits,
    v_ref_payload_bits,
    helper_data,
    hash_key_enroll,
    g, K, PAR, t, num_chunks,
    video_label,
):
    """
    Verify a probe against the enrolled reference.

    Stage 1 - Cosine gate: reject immediately if cosine < COSINE_GATE_THRESHOLD.
    Stage 2 - BCH fuzzy commitment: recover r and compare SHA-256 hash.
    """
    # Stage 1: cosine gate
    cosine = cosine_similarity(
        probe_embedding, enrolled_embedding,
        label=f"gate_{video_label}_vs_enrolled"
    )
    print(f"  Cosine (probe vs enrolled) : {cosine:.8f}  "
          f"(gate threshold = {COSINE_GATE_THRESHOLD})")

    if cosine < COSINE_GATE_THRESHOLD:
        print(f"  COSINE GATE : REJECTED - different person, BCH not run.")
        print("-" * 60)
        return {
            "label"         : video_label,
            "cosine"        : cosine,
            "gate_passed"   : False,
            "hamming_total" : None,
            "max_chunk"     : None,
            "chunks_over_t" : None,
            "failed_chunks" : None,
            "corrected"     : None,
            "remaining"     : None,
            "hash_matches"  : False,
        }

    print(f"  COSINE GATE : PASSED")
    print()

    # Stage 2: BCH fuzzy commitment
    pad_vx  = (num_chunks * K) - len(vx_payload_bits)
    pad_ref = (num_chunks * K) - len(v_ref_payload_bits)
    vx_ext  = list(vx_payload_bits)    + [0] * pad_vx
    ref_ext = list(v_ref_payload_bits) + [0] * pad_ref

    total_ham     = sum(a != b for a, b in zip(vx_payload_bits, v_ref_payload_bits))
    per_chunk_ham = [
        sum(ref_ext[i * K + j] != vx_ext[i * K + j] for j in range(K))
        for i in range(num_chunks)
    ]
    max_chunk_ham   = max(per_chunk_ham)
    chunks_over_t   = sum(1 for e in per_chunk_ham if e > t)
    chunks_within_t = num_chunks - chunks_over_t

    print(f"  Total bit differences   : {total_ham} / {len(vx_payload_bits)}"
          f"  ({total_ham / len(vx_payload_bits) * 100:.2f}%)")
    print(f"  Max errors in one chunk : {max_chunk_ham}  (BCH limit = {t})")
    print(f"  Chunks within  t={t}    : {chunks_within_t} / {num_chunks}")
    print(f"  Chunks exceeding t={t}  : {chunks_over_t}  / {num_chunks}")
    print()
    print("  Per-chunk Hamming distances:")
    for i in range(0, num_chunks, 8):
        row  = per_chunk_ham[i : i + 8]
        line = "  ".join(f"c{i+j:02d}:{row[j]:2d}" for j in range(len(row)))
        print(f"    {line}")

    recovered_r = []
    total_corr  = 0
    failed      = 0
    for i in range(num_chunks):
        vx_chunk  = vx_ext[i * K : (i + 1) * K]
        vx_pad    = vx_chunk + [0] * PAR
        noisy_cw  = [a ^ b for a, b in zip(helper_data[i], vx_pad)]
        r_hat, nerr = bch_decode(noisy_cw, g, K, PAR, t)
        if nerr >= 0:
            total_corr += nerr
            recovered_r.extend(r_hat)
        else:
            failed += 1
            recovered_r.extend(vx_chunk)

    hash_verify  = hashlib.sha256(bits_to_bytes(recovered_r)).hexdigest()
    hash_matches = hash_verify == hash_key_enroll

    remaining = sum(
        a != b for a, b in zip(recovered_r, ref_ext[:len(vx_payload_bits)])
    )

    print()
    print(f"  BCH errors corrected    : {total_corr}")
    print(f"  Failed chunks           : {failed}  (0 = full recovery)")
    print(f"  Remaining bit errors    : {remaining}")
    print()
    print(f"  Hash enrolled ref       : {hash_key_enroll}")
    print(f"  Hash {video_label} result : {hash_verify}")
    verdict = "PASS - SAME PERSON - hashes match" if hash_matches else "FAIL - REJECTED - hashes do NOT match"
    print(f"  Result                  : {verdict}")
    print("-" * 60)

    return {
        "label"         : video_label,
        "cosine"        : cosine,
        "gate_passed"   : True,
        "hamming_total" : total_ham,
        "max_chunk"     : max_chunk_ham,
        "chunks_over_t" : chunks_over_t,
        "failed_chunks" : failed,
        "corrected"     : total_corr,
        "remaining"     : remaining,
        "hash_matches"  : hash_matches,
    }


# ===========================================================================
# SECTION 10 - MAIN PIPELINE
# ===========================================================================

def run():
    sep = "=" * 70

    log.info(f"Building BCH(N={BCH_N}, t={BCH_T_DESIGNED}) generator polynomial ...")
    g, BCH_K, BCH_PAR = build_bch_generator(BCH_T_DESIGNED)

    PAYLOAD_BITS = 512 * QUANT_BITS
    NUM_CHUNKS   = math.ceil(PAYLOAD_BITS / BCH_K)
    T_TOTAL      = NUM_CHUNKS * BCH_T_DESIGNED
    PAD_NEEDED   = NUM_CHUNKS * BCH_K - PAYLOAD_BITS
    VIS_ROWS     = FACE_SIZE - int(FACE_SIZE * MASK_FRACTION)
    BLACK_ROWS   = FACE_SIZE - VIS_ROWS

    print(sep)
    print("  ADAFACE EMBEDDING PIPELINE + BCH FUZZY-COMMITMENT")
    print("  Phase 10  (QUANT_BITS=5, t=35)")
    print("  Enroll V4 (Android_NoBeard)  /  Verify V5 (Android_v5)")
    print(sep)
    for i, vp in enumerate(VIDEO_PATHS, 1):
        label = VIDEO_LABELS.get(f"video_{i}", f"video_{i}")
        print(f"    {i}. [{label}]  {Path(vp).name}")
    print(f"\n  Frame strategy  : top {FRAMES_TO_USE} sharpest from "
          f"{FRAMES_TO_USE * CANDIDATE_MULTIPLIER} candidates")
    print(f"  Mask            : rows {VIS_ROWS}..111 black  |  rows 0..{VIS_ROWS-1} visible")
    print(f"  BCH code        : BCH(N={BCH_N}, K={BCH_K}, t={BCH_T_DESIGNED})"
          f" x {NUM_CHUNKS} chunks  (t_total={T_TOTAL})")
    print(f"  Parity bits     : {BCH_PAR}  |  last chunk pad: {PAD_NEEDED} bits")
    print(f"  QUANT_BITS      : {QUANT_BITS}  ({2**QUANT_BITS} levels, payload={PAYLOAD_BITS} bits)")
    print(f"  Cosine gate     : {COSINE_GATE_THRESHOLD}")
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
        if result is not None:
            video_name, final_emb = result
            embeddings[video_name] = final_emb

    print(f"\n{sep}")
    print("  FINAL EMBEDDINGS")
    print(sep)
    for name, emb in embeddings.items():
        label = VIDEO_LABELS.get(name, name)
        print(f"  {name:10} [{label:20}]: norm={np.linalg.norm(emb):.8f}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep)

    similarities: Dict[str, float] = {}
    if len(embeddings) >= 2:
        similarities = compute_pairwise_similarities(embeddings)
        if similarities:
            analysis = analyze_similarities(similarities)
            print_similarity_analysis(analysis)
            print_v5_comparisons(similarities)
            print(f"  Overall quality: {analysis['quality_assessment']}")
    else:
        log.warning(f"Only {len(embeddings)} video(s) processed - need at least 2 for comparison.")

    # =======================================================================
    # BCH FUZZY-COMMITMENT: ENROLL V4, VERIFY V5
    # =======================================================================

    if "video_4" not in embeddings:
        log.error("video_4 unavailable - cannot enroll V4.")
        return embeddings, similarities
    if "video_5" not in embeddings:
        log.error("video_5 unavailable - cannot verify V5.")
        return embeddings, similarities

    print(f"\n{'=' * 70}")
    print("  PHASE 10 - BCH FUZZY-COMMITMENT  (ENROLL V4 / VERIFY V5)")
    print(f"{'=' * 70}")

    v4_bits, _, v4_min, v4_max = embedding_to_payload(embeddings["video_4"])
    log.info(f"V4 quantised - scale [{v4_min:.5f}, {v4_max:.5f}] | payload = {len(v4_bits)} bits")

    # Enrollment
    print(f"\n{'-' * 60}")
    print("  ENROLLMENT - V4 (Android_NoBeard)")
    print(f"{'-' * 60}")

    helper_data, hash_key_H4 = bch_enroll(
        v_ref_payload_bits=v4_bits,
        g=g, K=BCH_K, PAR=BCH_PAR,
        num_chunks=NUM_CHUNKS,
    )

    helper_hex = bits_to_hex([b for hd in helper_data for b in hd])
    print(f"  Hash key (SHA-256)   : {hash_key_H4}")
    print(f"  Helper (first 64 hex): {helper_hex[:64]}...")
    print(f"  V4 scale             : [{v4_min:.5f}, {v4_max:.5f}]")
    print(f"{'-' * 60}")

    # Verification
    print(f"\n{'-' * 60}")
    print("  VERIFICATION - V5 (Android_v5) against enrolled V4")
    print(f"{'-' * 60}")

    v5_bits, _, _, _ = embedding_to_payload(
        embeddings["video_5"], shared_min=v4_min, shared_max=v4_max
    )
    log.info(
        f"V5 quantised using V4 scale [{v4_min:.5f}, {v4_max:.5f}] | "
        f"payload = {len(v5_bits)} bits"
    )

    result_v5 = bch_verify(
        probe_embedding    = embeddings["video_5"],
        enrolled_embedding = embeddings["video_4"],
        vx_payload_bits    = v5_bits,
        v_ref_payload_bits = v4_bits,
        helper_data        = helper_data,
        hash_key_enroll    = hash_key_H4,
        g=g, K=BCH_K, PAR=BCH_PAR, t=BCH_T_DESIGNED,
        num_chunks         = NUM_CHUNKS,
        video_label        = "video_5",
    )

    # Summary table
    r = result_v5
    print(f"\n{'=' * 70}")
    print("  PHASE 10 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Enrollment  : V4 - Android_NoBeard  |  hash: {hash_key_H4}")
    print(f"  Scale       : [{v4_min:.5f}, {v4_max:.5f}]")
    print()

    cosine_str = f"{r['cosine']:.8f}" if r["cosine"] is not None else "N/A"
    gate_str   = "PASSED" if r["gate_passed"] else "FAILED"
    ham_str    = str(r["hamming_total"]) if r["hamming_total"] is not None else "N/A"
    max_str    = str(r["max_chunk"])     if r["max_chunk"]     is not None else "N/A"
    chk_str    = str(r["chunks_over_t"]) if r["chunks_over_t"] is not None else "N/A"
    fld_str    = str(r["failed_chunks"]) if r["failed_chunks"] is not None else "N/A"
    cor_str    = str(r["corrected"])     if r["corrected"]     is not None else "N/A"
    result_str = "PASS" if r["hash_matches"] else "FAIL"

    print(f"  {'Video':<12}  {'Cosine':>10}  {'Gate':>8}  {'Ham-Tot':>10}  "
          f"{'Max/Chk':>7}  {'Chk>t':>5}  {'Failed':>6}  {'Corrctd':>7}  {'Result':>6}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  "
          f"{'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")
    print(
        f"  {r['label']:<12}  {cosine_str:>10}  {gate_str:>8}  {ham_str:>10}  "
        f"{max_str:>7}  {chk_str:>5}  {fld_str:>6}  {cor_str:>7}  {result_str:>6}"
    )
    print(f"{'=' * 70}")
    return embeddings, similarities


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ADAFACE + BCH FUZZY-COMMITMENT  -  Phase 10  (QUANT_BITS=5, t=35)")
    print("  Enroll V4 (Android_NoBeard) / Verify V5 (Android_v5)")
    print("=" * 70)

    embeddings, pairwise_sims = run()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Videos processed : {len(embeddings)}/{len(VIDEO_PATHS)}")
    print(f"  Pairs compared   : {len(pairwise_sims)}")

    if pairwise_sims:
        sv = list(pairwise_sims.values())
        print(f"  Avg similarity   : {float(np.mean(sv)):.8f}")
        print(f"  Range            : {min(sv):.8f} - {max(sv):.8f}")

    print("=" * 70)
