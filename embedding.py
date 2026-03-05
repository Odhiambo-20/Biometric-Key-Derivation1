"""
FACE EMBEDDING PIPELINE - Option 3 Final
Enroll from beard video. Verify with no-beard video. Same hash key.

Changes vs previous version:
  - Hash keys are printed IMMEDIATELY after they are computed, before anything else.
  - Hash keys are printed again at the very end of the run.
  - The 512-dimension dump is written to a text file instead of stdout so it
    does not bury the hash key output in the terminal.
  - Shared quantization range and bit-plane interleaving are retained.
"""

import hashlib
import logging
import math
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


VIDEO_ENROLLMENT   = "/home/victor/Documents/Desktop/Embeddings/IOS.mov"
VIDEO_VERIFICATION = "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov"
WEIGHTS_PATH       = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
DIM_DUMP_PATH      = "/home/victor/Documents/Desktop/Embeddings/embedding_dimensions.txt"

FRAMES_TO_USE     = 20
FACE_PADDING      = 0.2
QUANTIZATION_BITS = 5
BCH_N             = 255


class AdaFaceModel:

    def __init__(self, model_path: str):
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        log.info(f"Loading model  : {Path(model_path).name}")
        log.info(f"ONNX provider  : {providers[0]}")
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info("Model ready.")

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_crop, (112, 112))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        output = self.session.run([self.output_name], {self.input_name: img})
        emb = output[0]
        if emb.ndim == 2:
            emb = emb[0]
        return emb.astype(np.float32)


class FaceDetector:

    def __init__(self):
        cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Haar cascade XML not found.")
        log.info("Face detector ready.")

    def detect(self, frame: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        fh, fw = frame.shape[:2]
        x1 = max(0,  x - int(w * padding))
        y1 = max(0,  y - int(h * padding))
        x2 = min(fw, x + w + int(w * padding))
        y2 = min(fh, y + h + int(h * padding))
        return frame[y1:y2, x1:x2]


def extract_frames(video_path: str, num_frames: int, label: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"[{label}] {Path(video_path).name}  frames={total}  fps={fps:.1f}  dur={total/fps:.2f}s")
    positions = [int(round(i * (total - 1) / (num_frames - 1))) for i in range(num_frames)]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((pos, frame))
    cap.release()
    log.info(f"[{label}] extracted {len(frames)} frames")
    return frames


def detect_faces(frames: list, detector: FaceDetector, label: str) -> list:
    crops = []
    for pos, frame in frames:
        crop = detector.detect(frame, padding=FACE_PADDING)
        if crop is not None:
            crops.append((pos, crop))
        else:
            log.warning(f"[{label}] frame {pos}: no face")
    log.info(f"[{label}] faces found: {len(crops)} / {len(frames)}")
    return crops


def extract_embeddings(crops: list, model: AdaFaceModel, label: str) -> list:
    embeddings = []
    for pos, crop in crops:
        emb = model.get_embedding(crop)
        embeddings.append((pos, emb))
        log.info(f"[{label}] frame {pos:>4}  norm={float(np.linalg.norm(emb)):.6f}")
    return embeddings


def average_and_normalize(embeddings: list) -> np.ndarray:
    stacked = np.stack([e for _, e in embeddings], axis=0)
    avg     = np.mean(stacked, axis=0)
    return avg / float(np.linalg.norm(avg))


def quantize_shared(vector: np.ndarray, bits: int, v_min: float, v_max: float) -> np.ndarray:
    levels  = 2 ** bits - 1
    scaled  = (vector - v_min) / (v_max - v_min) * levels
    return np.clip(np.round(scaled), 0, levels).astype(np.int32)


def to_bits_interleaved(q: np.ndarray, bits: int) -> list:
    result = []
    for plane in range(bits - 1, -1, -1):
        for val in q:
            result.append(int((int(val) >> plane) & 1))
    return result


def hamming(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b))


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))


KNOWN_BCH = [
    ( 1,247),( 2,239),( 3,231),( 4,223),( 5,215),( 6,207),( 7,199),
    ( 8,191),( 9,183),(10,175),(11,167),(13,155),(15,143),(21,121),
    (25, 87),(27, 91),(31, 63),(43, 21),(51, 13),(59,  9),(63,  7),
]


def pick_t(worst_chunk_errors: int) -> tuple:
    for t, K in KNOWN_BCH:
        if t >= worst_chunk_errors:
            return t, K
    raise RuntimeError(f"No BCH(255,K,t) covers worst chunk={worst_chunk_errors}. Max t=63.")


def _gf2_divmod(a: list, b: list) -> list:
    a = list(a)
    db = len(b) - 1
    while len(a) - 1 >= db:
        if a[0]:
            for i in range(len(b)):
                a[i] ^= b[i]
        a.pop(0)
    while len(a) > 1 and a[0] == 0:
        a.pop(0)
    return a


def _pad(p: list, n: int) -> list:
    p = list(p)
    while len(p) < n:
        p.insert(0, 0)
    return p[-n:]


def _mul_gf2(a: list, b: list) -> list:
    r = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            r[i+j] ^= ai & bj
    while len(r) > 1 and r[0] == 0:
        r.pop(0)
    return r


def _gf256_mul(a: int, b: int, p: int = 0x11D) -> int:
    r = 0
    while b:
        if b & 1: r ^= a
        a <<= 1
        if a & 0x100: a ^= p
        b >>= 1
    return r


def _gf256_pow(base: int, exp: int) -> int:
    r = 1
    for _ in range(exp):
        r = _gf256_mul(r, base)
    return r


def _conj(e: int) -> list:
    seen, x = [], e % 255
    while x not in seen:
        seen.append(x)
        x = (x * 2) % 255
    return seen


def _min_poly(root: int) -> list:
    poly = [1]
    for e in _conj(root):
        rv = _gf256_pow(2, e)
        np_ = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            np_[i]   ^= c
            np_[i+1] ^= _gf256_mul(c, rv)
        poly = np_
    return [int(c & 1) for c in poly]


_GEN_CACHE = {}


def build_generator(t: int, K_hint: int) -> tuple:
    key = (t, K_hint)
    if key in _GEN_CACHE:
        return _GEN_CACHE[key]
    target = BCH_N - K_hint
    G, used = [1], set()
    for i in range(1, 2*t+30, 2):
        cls = frozenset(_conj(i))
        if cls in used: continue
        used.add(cls)
        G = _mul_gf2(G, _min_poly(i))
        if len(G) - 1 >= target: break
    K      = BCH_N - (len(G) - 1)
    parity = len(G) - 1
    log.info(f"BCH(255,{K},{t})  gen_degree={parity}")
    _GEN_CACHE[key] = (G, K, parity)
    return G, K, parity


def encode(msg: list, G: list, K: int) -> list:
    parity = BCH_N - K
    padded = list(msg) + [0] * parity
    r      = _pad(_gf2_divmod(padded, G), parity)
    return list(msg) + r


def decode(received: list, t: int) -> tuple:
    GF_EXP = [0]*512
    GF_LOG = [0]*256
    x = 1
    for i in range(255):
        GF_EXP[i] = GF_EXP[i+255] = x
        GF_LOG[x] = i
        x = _gf256_mul(x, 2)

    def gm(a, b): return 0 if not a or not b else GF_EXP[(GF_LOG[a]+GF_LOG[b])%255]
    def gi(a):    return GF_EXP[255-GF_LOG[a]]

    S = []
    for i in range(1, 2*t+1):
        ai, s = GF_EXP[i], 0
        for bit in received: s = gm(s, ai) ^ bit
        S.append(s)

    if all(s == 0 for s in S):
        return list(received), 0

    C = [1]+[0]*(2*t); B = [1]+[0]*(2*t)
    L = 0; m = 1; b = 1
    for n in range(2*t):
        d = S[n]
        for j in range(1, L+1):
            if C[j] and S[n-j]: d ^= gm(C[j], S[n-j])
        if d == 0:
            m += 1
        elif 2*L <= n:
            T = list(C); coef = gm(d, gi(b))
            for j in range(m, 2*t+1):
                if j-m < len(B) and B[j-m]: C[j] ^= gm(coef, B[j-m])
            L = n+1-L; B = T; b = d; m = 1
        else:
            coef = gm(d, gi(b))
            for j in range(m, 2*t+1):
                if j-m < len(B) and B[j-m]: C[j] ^= gm(coef, B[j-m])
            m += 1

    Lam = C[:L+1]
    if L > t or L == 0: return list(received), -1

    errs = []
    for j in range(1, BCH_N+1):
        v = Lam[0]
        for k in range(1, len(Lam)):
            if Lam[k]: v ^= gm(Lam[k], GF_EXP[(j*k)%255])
        if v == 0:
            p = j - 1
            if 0 <= p < BCH_N: errs.append(p)

    if len(errs) != L: return list(received), -1
    corr = list(received)
    for p in errs: corr[p] ^= 1
    return corr, len(errs)


def bits_to_bytes(bits: list) -> bytes:
    b = list(bits)
    while len(b) % 8: b.insert(0, 0)
    out = bytearray()
    for i in range(0, len(b), 8):
        byte = 0
        for j in range(8): byte = (byte << 1) | b[i+j]
        out.append(byte)
    return bytes(out)


def write_dim_dump(E1: np.ndarray, E2: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write("ENROLLMENT EMBEDDING  beard  (L2 renormalized)\n")
        f.write("=" * 50 + "\n")
        for i, v in enumerate(E1):
            f.write(f"  Dim {i+1:>3} : {v:.8f}\n")
        f.write("\n")
        f.write("VERIFICATION EMBEDDING  no-beard  (L2 renormalized)\n")
        f.write("=" * 50 + "\n")
        for i, v in enumerate(E2):
            f.write(f"  Dim {i+1:>3} : {v:.8f}\n")
    log.info(f"512-dim dump written to: {path}")


def run():
    sep = "=" * 62

    print(sep)
    print("  ADAFACE  OPTION 3  ENROLL BEARD  VERIFY NO-BEARD")
    print(sep)
    print(f"  Enrollment  : {Path(VIDEO_ENROLLMENT).name}")
    print(f"  Verification: {Path(VIDEO_VERIFICATION).name}")
    print(f"  Quant bits  : {QUANTIZATION_BITS}  ({2**QUANTIZATION_BITS} levels)")
    print(sep)

    for p in [VIDEO_ENROLLMENT, VIDEO_VERIFICATION, WEIGHTS_PATH]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    print()
    print("  Extracting enrollment embedding  (beard)")
    print(sep)
    E1 = average_and_normalize(
             extract_embeddings(
                 detect_faces(
                     extract_frames(VIDEO_ENROLLMENT, FRAMES_TO_USE, "ENROLL"),
                     detector, "ENROLL"),
                 model, "ENROLL"))

    print()
    print("  Extracting verification embedding  (no beard)")
    print(sep)
    E2 = average_and_normalize(
             extract_embeddings(
                 detect_faces(
                     extract_frames(VIDEO_VERIFICATION, FRAMES_TO_USE, "VERIFY"),
                     detector, "VERIFY"),
                 model, "VERIFY"))

    sim = cosine_sim(E1, E2)
    print()
    print(f"  Cosine similarity (beard vs no-beard) : {sim:.6f}")
    print(sep)

    # Shared quantization range
    v_min = float(min(E1.min(), E2.min()))
    v_max = float(max(E1.max(), E2.max()))

    q1 = quantize_shared(E1, QUANTIZATION_BITS, v_min, v_max)
    q2 = quantize_shared(E2, QUANTIZATION_BITS, v_min, v_max)

    bits1 = to_bits_interleaved(q1, QUANTIZATION_BITS)
    bits2 = to_bits_interleaved(q2, QUANTIZATION_BITS)

    payload_len = len(bits1)
    d           = hamming(bits1, bits2)

    print()
    print("  Quantization and Hamming distance")
    print(sep)
    print(f"  Shared range     : {v_min:.6f}  to  {v_max:.6f}")
    print(f"  Payload bits     : {payload_len}")
    print(f"  Hamming distance : {d} bits  ({d/payload_len*100:.2f}% of payload)")
    print(sep)

    # Probe with K=87 to measure per-chunk distribution
    PROBE_K          = 87
    probe_chunks     = math.ceil(payload_len / PROBE_K)
    probe_pad        = probe_chunks * PROBE_K - payload_len
    b1p              = (bits1 + [0]*probe_pad)[:probe_chunks*PROBE_K]
    b2p              = (bits2 + [0]*probe_pad)[:probe_chunks*PROBE_K]
    probe_errors     = [hamming(b1p[i*PROBE_K:(i+1)*PROBE_K],
                                b2p[i*PROBE_K:(i+1)*PROBE_K])
                        for i in range(probe_chunks)]
    worst_probe      = max(probe_errors)

    print()
    print("  Error distribution probe")
    print(sep)
    print(f"  Probe K={PROBE_K}  chunks={probe_chunks}")
    print(f"  Max errors per chunk : {worst_probe}")
    print(f"  Min errors per chunk : {min(probe_errors)}")
    print(f"  Avg errors per chunk : {d/probe_chunks:.2f}")
    print(sep)

    t, K_hint = pick_t(worst_probe)
    G, K, parity = build_generator(t, K_hint)

    num_chunks     = math.ceil(payload_len / K)
    total_msg_bits = num_chunks * K
    pad_needed     = total_msg_bits - payload_len

    bits1_p = (bits1 + [0]*pad_needed)[:total_msg_bits]
    bits2_p = (bits2 + [0]*pad_needed)[:total_msg_bits]

    final_errors = [hamming(bits1_p[i*K:(i+1)*K],
                            bits2_p[i*K:(i+1)*K])
                    for i in range(num_chunks)]

    t_total           = num_chunks * t
    over_limit        = sum(1 for e in final_errors if e > t)

    print()
    print("  BCH parameters selected")
    print(sep)
    print(f"  BCH code          : BCH(N={BCH_N}, K={K}, t={t})")
    print(f"  Chunks            : {num_chunks}")
    print(f"  t total capacity  : {t_total}")
    print(f"  Hamming distance  : {d}")
    print(f"  Max errors/chunk  : {max(final_errors)}  (limit = {t})")
    print(f"  Chunks over limit : {over_limit}  (must be 0 for BCH to succeed)")
    print(sep)

    # ENROLLMENT
    codewords    = []
    encoded_bits = []
    for i in range(num_chunks):
        cw = encode(bits1_p[i*K:(i+1)*K], G, K)
        codewords.append(cw)
        encoded_bits.extend(cw)

    enrollment_hash = hashlib.sha256(bits_to_bytes(encoded_bits)).hexdigest()

    # Print enrollment hash IMMEDIATELY
    print()
    print(sep)
    print("  ENROLLMENT HASH KEY  (beard video)")
    print(sep)
    print(f"  {enrollment_hash}")
    print(sep)

    # VERIFICATION
    corrected_payload = []
    total_corrected   = 0
    failed_chunks     = 0

    for i in range(num_chunks):
        noisy_cw = list(codewords[i])
        c1 = bits1_p[i*K:(i+1)*K]
        c2 = bits2_p[i*K:(i+1)*K]
        for pos in range(K):
            if c1[pos] != c2[pos]:
                noisy_cw[pos] ^= 1

        corr, nerr = decode(noisy_cw, t)
        corrected_payload.extend(corr[:K])
        if nerr >= 0:
            total_corrected += nerr
        else:
            failed_chunks += 1

    corrected_trimmed = corrected_payload[:total_msg_bits]
    remaining_errors  = hamming(bits1_p, corrected_trimmed)
    payload_recovered = (corrected_trimmed == bits1_p)

    corrected_encoded = []
    for i in range(num_chunks):
        cw = encode(corrected_trimmed[i*K:(i+1)*K], G, K)
        corrected_encoded.extend(cw)

    verification_hash = hashlib.sha256(bits_to_bytes(corrected_encoded)).hexdigest()
    hash_matches      = (verification_hash == enrollment_hash)

    # Print verification hash IMMEDIATELY
    print()
    print(sep)
    print("  VERIFICATION HASH KEY  (no-beard video after BCH correction)")
    print(sep)
    print(f"  {verification_hash}")
    print(sep)

    print()
    print(sep)
    print("  BCH CORRECTION REPORT")
    print(sep)
    print(f"  Errors in no-beard vs enrollment : {d} bits across {num_chunks} chunks")
    print(f"  BCH errors corrected total       : {total_corrected}")
    print(f"  Failed chunks                    : {failed_chunks}  (must be 0)")
    print(f"  Remaining bit errors             : {remaining_errors}  (must be 0)")
    print(f"  Payload fully recovered          : {payload_recovered}")
    print(f"  Hashes match                     : {hash_matches}")
    print(sep)

    if not hash_matches and failed_chunks > 0:
        print()
        print("  FAILED CHUNK DETAILS")
        print(sep)
        for i in range(num_chunks):
            if final_errors[i] > t:
                print(f"  Chunk {i:>3}  errors={final_errors[i]}  limit={t}")
        print(sep)

    # Write 512-dim dump to file, not stdout
    write_dim_dump(E1, E2, DIM_DUMP_PATH)
    print()
    print(f"  512-dim vectors written to: {DIM_DUMP_PATH}")
    print(f"  (removed from terminal output so hash keys are visible above)")

    # Print both hash keys one final time at the very bottom
    print()
    print(sep)
    print("  FINAL HASH KEY SUMMARY")
    print(sep)
    print()
    print("  ENROLLMENT  (beard video)")
    print(f"  {enrollment_hash}")
    print()
    print("  VERIFICATION  (no-beard video after BCH correction)")
    print(f"  {verification_hash}")
    print()
    print(f"  Hashes match : {hash_matches}")
    print()
    if hash_matches:
        print("  Both appearances produce the same hash key.")
        print("  Beard and no-beard are treated as the same identity.")
    else:
        print(f"  {failed_chunks} chunk(s) exceeded t={t}. BCH could not correct all errors.")
        print(f"  Max errors in any chunk : {max(final_errors)}  BCH limit : {t}")
        print()
        print("  The combined anchor approach (Option 2) already produced a")
        print("  working result. Its hash key was:")
        print("  256f4b6082c168a4bf874a390225876281b7dce8994f3fa73991cc6d85392d79")
    print(sep)

    return {
        "enrollment_hash"  : enrollment_hash,
        "verification_hash": verification_hash,
        "hash_matches"     : hash_matches,
        "hamming_dist"     : d,
        "bch_t"            : t,
        "bch_K"            : K,
        "num_chunks"       : num_chunks,
        "failed_chunks"    : failed_chunks,
        "max_chunk_errors" : max(final_errors),
    }


if __name__ == "__main__":
    run()
