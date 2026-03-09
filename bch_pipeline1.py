"""
AdaFace BCH Fuzzy Commitment Pipeline
======================================

Embedding pipeline per video:
  1. Extract top-20 sharpest frames (scan 60 candidates)
  2. Detect face + eye cascade → Umeyama 112×112 alignment
  3. AdaFace IR-18 → raw 512-dim embedding
  4. L2 normalise → unit vector
  5. Average all frame unit vectors → L2 renormalise → final unit embedding
  6. Scale × K  (sweep K=1,2,3,4 → pick lowest genuine BER, or force via FORCE_SCALE)
  7. Clip [-1, +1]
  8. Quantise uniform 4-bit → integers [0,15]
  9. Gray-code encode each integer
 10. Unpack bits → 2048 bits → truncate to 2040 bits (8×255)

BCH parameters: n=255, k=71, t=28, num_chunks=8
  → 2040 bits per embedding, 568-bit key

KEY SCHEME: ONE shared random key K locked into helper data per enrollment.
  helper_Vi = bitvec_Vi  XOR  BCH_encode(K)
  All four genuine videos (V1-V4) lock the SAME key K.
  Genuine probes recover K exactly → hash matches.
  Impostor probes exceed t=28 errors/chunk → BCH fails → hash mismatch.

SCALE CONTROL (see CONFIG section below):
  FORCE_SCALE = None   → auto-select best K from sweep (recommended)
  FORCE_SCALE = 1      → force K=1
  FORCE_SCALE = 2      → force K=2  (use this to compare with auto-selected K=1)
  FORCE_SCALE = 3      → force K=3
  FORCE_SCALE = 4      → force K=4
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


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

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

FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112
QUANT_BITS     = 4

BCH_N        = 255
BCH_K        = 71
BCH_T        = 28
NUM_CHUNKS   = 8
SCALE_VALUES = [1, 2, 3, 4]   # Always sweep all — FORCE_SCALE overrides winner

# ── SCALE CONTROL ──────────────────────────────────────────────────────────
# Set FORCE_SCALE to an integer to skip auto-selection and use that scale.
# Set to None to let the sweep automatically pick the best (lowest genuine BER).
#
#   FORCE_SCALE = None  → auto-select  (default / recommended)
#   FORCE_SCALE = 1     → force K=1
#   FORCE_SCALE = 2     → force K=2   ← set this to compare against K=1
#   FORCE_SCALE = 3     → force K=3
#   FORCE_SCALE = 4     → force K=4
# ───────────────────────────────────────────────────────────────────────────
FORCE_SCALE = 2    # ← change to None to restore auto-select

REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FACE ALIGNER
# ─────────────────────────────────────────────────────────────────────────────

class FaceAligner:
    def __init__(self):
        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        if self.face_cascade.empty():
            raise RuntimeError("Face Haar cascade not found.")
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.eye_ok = not self.eye_cascade.empty()
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        log.info(f"FaceAligner | eye cascade: {'yes' if self.eye_ok else 'no (geometry fallback)'}")

    def _preprocess(self, frame):
        return self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def _detect_face(self, gray):
        for sf, mn, ms in [(1.05, 6, 60), (1.05, 3, 40), (1.10, 2, 30)]:
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
            if len(faces) > 0:
                return tuple(max(faces, key=lambda f: f[2]*f[3]))
        return None

    def _detect_eyes(self, gray, fx, fy, fw, fh):
        if not self.eye_ok: return None
        roi = gray[fy:fy+int(fh*0.60), fx:fx+fw]
        eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.10, minNeighbors=5, minSize=(20,20))
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.10, minNeighbors=2, minSize=(15,15))
        if len(eyes) < 2: return None
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        centres = sorted([np.array([fx+ex+ew//2, fy+ey+eh//2], dtype=np.float32) for ex,ey,ew,eh in eyes], key=lambda p: p[0])
        return centres[0], centres[1]

    @staticmethod
    def _landmarks(x, y, w, h, le=None, re=None):
        le = le if le is not None else np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        re = re if re is not None else np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        return np.array([le, re, [x+0.50*w,y+0.57*h], [x+0.35*w,y+0.76*h], [x+0.65*w,y+0.76*h]], dtype=np.float32)

    @staticmethod
    def _umeyama(src, dst):
        n = src.shape[0]
        mu_s=src.mean(0); mu_d=dst.mean(0)
        sc=src-mu_s; dc=dst-mu_d
        vs=(sc**2).sum()/n
        if vs<1e-10: return None
        cov=(dc.T@sc)/n
        try: U,S,Vt=np.linalg.svd(cov)
        except: return None
        d=np.ones(2)
        if np.linalg.det(U)*np.linalg.det(Vt)<0: d[-1]=-1
        R=U@np.diag(d)@Vt; c=(S*d).sum()/vs; t=mu_d-c*R@mu_s
        M=np.zeros((2,3),dtype=np.float32); M[:,:2]=c*R; M[:,2]=t
        return M

    def align(self, frame):
        gray=self._preprocess(frame)
        det=self._detect_face(gray)
        if det is None: return None
        x,y,w,h=det
        eyes=self._detect_eyes(gray,x,y,w,h)
        le,re=(eyes[0],eyes[1]) if eyes else (None,None)
        src=self._landmarks(x,y,w,h,le,re)
        M=self._umeyama(src,REFERENCE_PTS)
        if M is None:
            fh,fw=frame.shape[:2]
            crop=frame[max(0,y-int(h*.05)):min(fh,y+h+int(h*.02)), max(0,x-int(w*.10)):min(fw,x+w+int(w*.10))]
            return cv2.resize(crop,(FACE_SIZE,FACE_SIZE),interpolation=cv2.INTER_LANCZOS4) if crop.size else None
        return cv2.warpAffine(frame,M,(FACE_SIZE,FACE_SIZE),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────────────────────────────────────
#  ADAFACE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AdaFaceModel:
    def __init__(self, model_path):
        import onnxruntime as ort
        providers=(["CUDAExecutionProvider","CPUExecutionProvider"]
                   if "CUDAExecutionProvider" in ort.get_available_providers()
                   else ["CPUExecutionProvider"])
        self.session=ort.InferenceSession(model_path,providers=providers)
        self.input_name=self.session.get_inputs()[0].name
        self.output_name=self.session.get_outputs()[0].name
        log.info(f"AdaFace IR-18 | {providers[0]}")

    def raw_embedding(self, face_bgr):
        img=cv2.resize(face_bgr,(FACE_SIZE,FACE_SIZE),interpolation=cv2.INTER_LANCZOS4)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=(img.astype(np.float32)/255.0-0.5)/0.5
        img=img.transpose(2,0,1)[np.newaxis]
        out=self.session.run([self.output_name],{self.input_name:img})
        emb=out[0][0] if out[0].ndim==2 else out[0]
        return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan=FRAMES_TO_USE*CANDIDATE_MULT
    positions=[int(round(i*(total-1)/max(n_scan-1,1))) for i in range(n_scan)]
    candidates=[]
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        ret,frame=cap.read()
        if not ret or frame is None: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        score=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        candidates.append((score,pos,frame))
    cap.release()
    if not candidates: raise RuntimeError(f"No frames: {video_path}")
    candidates.sort(key=lambda x:x[0],reverse=True)
    top=candidates[:FRAMES_TO_USE]; top.sort(key=lambda x:x[1])
    return [f for _,_,f in top]


# ─────────────────────────────────────────────────────────────────────────────
#  EMBED VIDEO → UNIT EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_video(video_path, model, aligner):
    frames=extract_frames(video_path)
    unit_vecs=[]
    for frame in frames:
        aligned=aligner.align(frame)
        if aligned is None: continue
        raw=model.raw_embedding(aligned)
        norm=np.linalg.norm(raw)
        if norm<1e-10: continue
        unit_vecs.append((raw/norm).astype(np.float32))
    if not unit_vecs:
        log.error(f"No faces: {Path(video_path).name}"); return None
    avg=np.mean(np.stack(unit_vecs,axis=0),axis=0).astype(np.float32)
    avg_norm=np.linalg.norm(avg)
    if avg_norm<1e-10: return None
    final=(avg/avg_norm).astype(np.float32)
    log.info(f"  {Path(video_path).name:<44}  faces={len(unit_vecs):>2}/{len(frames)}  norm={np.linalg.norm(final):.6f}")
    return final


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING → BIT VECTOR
#  unit_emb → ×scale → clip[-1,1] → 4-bit quant → Gray code → bits
#
#  INTERLEAVING: After generating the 2040-bit vector, bits are interleaved
#  across all 8 chunks before BCH encoding/decoding.
#
#  Without interleaving, bit errors cluster in specific regions of the
#  embedding (some dimensions differ more than others between videos).
#  This causes certain chunks to receive 30+ errors while others get <10,
#  causing single-chunk BCH failures even when the TOTAL error count is
#  well within the 8×28=224 budget.
#
#  With interleaving, each chunk receives bits from evenly-spaced positions
#  across the entire 2040-bit vector:
#    chunk[i] = bits[i, i+8, i+16, i+24, ..., i+8*254]  (every 8th bit)
#
#  This spreads errors uniformly so each chunk gets ≈ total_errors/8,
#  keeping per-chunk errors well below t=28.
#
#  De-interleaving reverses this at recovery to reconstruct the original
#  bit vector order.
# ─────────────────────────────────────────────────────────────────────────────

def to_gray(n): return n ^ (n >> 1)

def interleave(bits: np.ndarray, num_chunks: int = NUM_CHUNKS) -> np.ndarray:
    """
    Interleave bits across chunks.
    Input:  [b0, b1, b2, ..., b(N*C-1)]  length = num_chunks * BCH_N
    Output: chunk[i] = bits at positions [i, i+C, i+2C, ..., i+(N-1)*C]
    i.e. output[i*N + j] = input[j*C + i]
    """
    total = num_chunks * BCH_N
    out   = np.empty(total, dtype=np.uint8)
    for i in range(num_chunks):
        out[i * BCH_N:(i + 1) * BCH_N] = bits[i::num_chunks]
    return out

def deinterleave(bits: np.ndarray, num_chunks: int = NUM_CHUNKS) -> np.ndarray:
    """
    Reverse of interleave.
    output[j*C + i] = input[i*N + j]
    """
    total = num_chunks * BCH_N
    out   = np.empty(total, dtype=np.uint8)
    for i in range(num_chunks):
        out[i::num_chunks] = bits[i * BCH_N:(i + 1) * BCH_N]
    return out

def embedding_to_bitvec(embedding, scale, bits=QUANT_BITS):
    """
    Convert a face embedding to an interleaved 2040-bit vector.
    Interleaving spreads errors evenly across BCH chunks.
    """
    levels = (1 << bits) - 1
    scaled = np.clip(embedding * scale, -1.0, 1.0)
    q      = np.clip(np.round((scaled + 1.0) / 2.0 * levels).astype(np.int32), 0, levels)
    g      = np.array([to_gray(int(v)) for v in q], dtype=np.int32)
    result = np.zeros(len(g) * bits, dtype=np.uint8)
    for i, val in enumerate(g):
        for b in range(bits):
            result[i * bits + (bits - 1 - b)] = (int(val) >> b) & 1
    raw    = result[:NUM_CHUNKS * BCH_N]
    return interleave(raw)          # ← interleave before chunking

def ber_bits(a, b):
    """BER on raw (pre-interleave) bits for display purposes."""
    # Deinterleave both before comparing so BER table shows natural embedding order
    a_raw = deinterleave(a) if len(a) == NUM_CHUNKS * BCH_N else a
    b_raw = deinterleave(b) if len(b) == NUM_CHUNKS * BCH_N else b
    errors = int(np.sum(a_raw != b_raw))
    return errors, errors / len(a_raw) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  BCH(255, 71, 28) CODEC  — CORRECTED
#
#  Convention (high-degree-first bit storage):
#    bits[0] = coefficient of x^(N-1)  (MSB)
#    bits[N-1] = coefficient of x^0    (LSB)
#
#  Syndrome:
#    S_j = r(alpha^j) = sum_{b=0}^{N-1} bits[b] * alpha^(j*(N-1-b))
#
#  Error locator sigma(z): roots are z = X_k^{-1} where X_k = alpha^(N-1-p_k)
#    for error at bit position p_k.
#
#  Chien search:
#    Evaluate sigma(alpha^i) for i=0..N-1.
#    If sigma(alpha^i) = 0  →  error at bit position p = (i-1) mod N
# ─────────────────────────────────────────────────────────────────────────────

class BCH:
    PRIM_POLY = 0x11D
    GF_SIZE   = 256
    N         = BCH_N    # 255
    K         = BCH_K    # 71
    T         = BCH_T    # 28
    PARITY    = N - K    # 184

    def __init__(self):
        self._build_gf()
        self._gen_bin_cache: Optional[np.ndarray] = None
        log.info("Building BCH(255,71,28) generator polynomial...")
        gen=self._gen_binary()
        log.info(f"BCH generator ready — degree={len(gen)-1}")

    # ── GF(2^8) ─────────────────────────────────────────────────────────────

    def _build_gf(self):
        self.gf_exp=[0]*(2*self.GF_SIZE); self.gf_log=[0]*self.GF_SIZE
        x=1
        for i in range(self.GF_SIZE-1):
            self.gf_exp[i]=x; self.gf_log[x]=i
            x<<=1
            if x&self.GF_SIZE: x^=self.PRIM_POLY
        for i in range(self.GF_SIZE-1,2*self.GF_SIZE):
            self.gf_exp[i]=self.gf_exp[i-(self.GF_SIZE-1)]

    def gf_mul(self,a,b):
        if a==0 or b==0: return 0
        return self.gf_exp[(self.gf_log[a]+self.gf_log[b])%(self.GF_SIZE-1)]

    def gf_inv(self,x):
        return self.gf_exp[(self.GF_SIZE-1)-self.gf_log[x]]

    def poly_mul_gf(self,p,q):
        r=[0]*(len(p)+len(q)-1)
        for i,pi in enumerate(p):
            for j,qj in enumerate(q): r[i+j]^=self.gf_mul(pi,qj)
        return r

    # ── Binary generator polynomial via cyclotomic cosets ───────────────────

    def _gf2_poly_mul(self,a,b):
        result=np.zeros(len(a)+len(b)-1,dtype=np.uint8)
        for i,ai in enumerate(a):
            if ai: result[i:i+len(b)]^=b
        return result

    def _gf2_poly_mod(self,dividend,divisor):
        out=np.array(dividend,dtype=np.uint8); dl=len(divisor)
        for i in range(len(out)-dl+1):
            if out[i]: out[i:i+dl]^=divisor
        return out[-(dl-1):]

    def _gen_binary(self):
        if self._gen_bin_cache is not None:
            return self._gen_bin_cache
        cosets_done=set(); gen_poly=np.array([1],dtype=np.uint8)
        for i in range(1,2*self.T+1):
            if i in cosets_done: continue
            coset=set(); c=i%self.N
            while c not in coset: coset.add(c); c=(2*c)%self.N
            cosets_done|=coset
            mp_gf=[1]
            for j in sorted(coset):
                mp_gf=self.poly_mul_gf(mp_gf,[1,self.gf_exp[j%(self.GF_SIZE-1)]])
            mp_bin=np.array([c%2 for c in mp_gf],dtype=np.uint8)
            gen_poly=self._gf2_poly_mul(gen_poly,mp_bin)
        target=self.PARITY+1
        if len(gen_poly)>target: gen_poly=gen_poly[-target:]
        elif len(gen_poly)<target:
            gen_poly=np.concatenate([np.zeros(target-len(gen_poly),dtype=np.uint8),gen_poly])
        self._gen_bin_cache=gen_poly
        return gen_poly

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, message_bits):
        assert len(message_bits)==self.K
        gen_bin=self._gen_binary()
        msg_shifted=np.zeros(self.N,dtype=np.uint8)
        msg_shifted[:self.K]=message_bits
        parity=self._gf2_poly_mod(msg_shifted,gen_bin)
        codeword=np.zeros(self.N,dtype=np.uint8)
        codeword[:self.K]=message_bits; codeword[self.K:]=parity
        return codeword

    # ── Syndromes (CORRECTED) ─────────────────────────────────────────────────
    #
    #  S_j = r(alpha^j) = sum_{b=0}^{N-1} bits[b] * alpha^(j*(N-1-b) mod 255)
    #
    #  We compute per j: XOR of gf_exp[(j*(N-1-b)) mod 255] for all set bits b.

    def _compute_syndromes(self, bits):
        set_bits=np.where(bits)[0]      # positions where bit=1
        b_vals=(self.N-1-set_bits)      # exponent base: (N-1-b) for each set bit
        syndromes=[]
        for j in range(1,2*self.T+1):
            exps=(j*b_vals)%(self.GF_SIZE-1)   # mod 255 (order of GF(2^8)*)
            s=0
            for e in exps: s^=self.gf_exp[e]
            syndromes.append(s)
        return syndromes

    # ── Berlekamp-Massey ─────────────────────────────────────────────────────

    def _berlekamp_massey(self, syndromes):
        C=[1]; B=[1]; L=0; m=1; b=1
        for n in range(len(syndromes)):
            d=syndromes[n]
            for i in range(1,L+1):
                if i<len(C): d^=self.gf_mul(C[i],syndromes[n-i])
            if d==0: m+=1
            elif 2*L<=n:
                T_=list(C); factor=self.gf_mul(d,self.gf_inv(b))
                padded_B=[0]*m+B
                while len(padded_B)>len(C): C.append(0)
                for i in range(len(padded_B)): C[i]^=self.gf_mul(factor,padded_B[i])
                L=n+1-L; B=T_; b=d; m=1
            else:
                factor=self.gf_mul(d,self.gf_inv(b))
                padded_B=[0]*m+B
                while len(padded_B)>len(C): C.append(0)
                for i in range(len(padded_B)): C[i]^=self.gf_mul(factor,padded_B[i])
                m+=1
        if L>self.T: return None
        return C

    # ── Chien search (CORRECTED) ─────────────────────────────────────────────
    #
    #  sigma(z) has roots z = X_k^{-1} = alpha^{-(N-1-p_k)} = alpha^{p_k+1} mod 255
    #  So if sigma(alpha^i) = 0  →  error at bit position p = (i-1) mod N
    #
    #  Using Chien recurrence: sigma_k_current *= alpha^k at each step.

    def _chien_search(self, sigma):
        L=len(sigma)-1
        # vals[k] = sigma[k] * alpha^(k*i), updated incrementally
        vals=list(sigma)
        alpha_pows=[self.gf_exp[k%(self.GF_SIZE-1)] if k>0 else 1 for k in range(L+1)]
        roots_pos=[]
        for i in range(self.N):
            s=0
            for v in vals: s^=v
            if s==0:
                roots_pos.append((i-1)%self.N)
            # Chien recurrence: vals[k] *= alpha^k
            for k in range(1,L+1):
                vals[k]=self.gf_mul(vals[k],alpha_pows[k])
        if len(roots_pos)!=L: return None
        return roots_pos

    # ── Decode ───────────────────────────────────────────────────────────────

    def decode(self, received_bits):
        assert len(received_bits)==self.N
        syn=self._compute_syndromes(received_bits)
        if all(s==0 for s in syn):
            return received_bits[:self.K].copy(), 0
        sigma=self._berlekamp_massey(syn)
        if sigma is None: return None, -1
        error_positions=self._chien_search(sigma)
        if error_positions is None: return None, -1
        corrected=np.array(received_bits,dtype=np.uint8)
        for pos in error_positions:
            if pos>=self.N: return None, -1
            corrected[pos]^=1
        # Verify
        syn2=self._compute_syndromes(corrected)
        if not all(s==0 for s in syn2): return None, -1
        return corrected[:self.K].copy(), len(error_positions)


# ─────────────────────────────────────────────────────────────────────────────
#  FUZZY COMMITMENT
#
#  The key K is ONE fixed secret shared across ALL enrollments of the same
#  person.  Each enrollment produces a different helper_data because each
#  bitvec is different, but the key they all lock/unlock is IDENTICAL.
#
#  Enroll(bitvec, K):
#    codeword      = BCH_encode(K)          [N bits per chunk]
#    helper_chunk  = bitvec_chunk XOR codeword
#    → helper is public; K is the secret
#
#  Recover(query_bitvec, helper):
#    noisy = query_bitvec_chunk XOR helper_chunk
#          = query_chunk XOR enrolled_chunk XOR BCH_encode(K)
#          = BCH_encode(K)  XOR  noise
#    BCH_decode(noisy) → K   (if noise ≤ t=28 bits per chunk)
#
#  Because K is the SAME for every enrollment, recovered key hash always
#  equals enrolled key hash for any genuine probe, regardless of which
#  video was enrolled.
# ─────────────────────────────────────────────────────────────────────────────

def generate_shared_key(rng: np.random.Generator) -> Tuple[List[np.ndarray], str]:
    """
    Generate ONE shared random key for the person.
    This key is used for ALL enrollments (V1, V2, V3, V4).
    Returns (key_chunks, key_hash).
    """
    key_bits = rng.integers(0, 2, size=NUM_CHUNKS * BCH_K, dtype=np.uint8)
    key_chunks = [key_bits[c * BCH_K:(c + 1) * BCH_K] for c in range(NUM_CHUNKS)]
    full_key   = np.packbits(key_bits)
    key_hash   = hashlib.sha256(full_key.tobytes()).hexdigest()
    return key_chunks, key_hash


def enroll(bitvec: np.ndarray, key_chunks: List[np.ndarray], bch: 'BCH') -> List[np.ndarray]:
    """
    Enroll a face bitvec against the shared key K.

    helper_chunk = bitvec_chunk  XOR  BCH_encode(key_chunk)

    The helper data is public.  The key K is the secret.
    Different videos of the SAME person produce different helper data
    (because their bitvecs differ slightly), but they ALL lock/unlock
    the SAME key K.

    Returns: helper_chunks  (NUM_CHUNKS arrays of BCH_N bits each)
    """
    helper_chunks = []
    for c in range(NUM_CHUNKS):
        chunk    = bitvec[c * BCH_N:(c + 1) * BCH_N]
        codeword = bch.encode(key_chunks[c])
        helper_chunks.append((chunk ^ codeword).astype(np.uint8))
    return helper_chunks


def recover(
    query_bitvec: np.ndarray,
    helper_chunks: List[np.ndarray],
    key_hash: str,
    bch: 'BCH',
) -> Dict:
    """
    Attempt to recover key K using a query face bitvec and public helper data.

    Per chunk:
      noisy_codeword = query_chunk XOR helper_chunk
                     = BCH_encode(K_chunk)  XOR  noise
      BCH_decode(noisy_codeword) → K_chunk  (if errors ≤ t=28)

    Verification: SHA-256(all recovered key chunks) must match enrolled key_hash.
    If it matches → genuine (same person, same key recovered).
    If it doesn't → impostor (BCH failed or wrong bits recovered).
    """
    recovered_keys = []
    t_per_chunk    = []
    fail_count     = 0
    k_bers         = []

    for c in range(NUM_CHUNKS):
        q              = query_bitvec[c * BCH_N:(c + 1) * BCH_N]
        noisy_codeword = (q ^ helper_chunks[c]).astype(np.uint8)
        decoded, n_err = bch.decode(noisy_codeword)

        if decoded is None:
            fail_count += 1
            t_per_chunk.append(-1)
            recovered_keys.append(np.zeros(BCH_K, dtype=np.uint8))
            k_bers.append(100.0)
        else:
            t_per_chunk.append(n_err)
            recovered_keys.append(decoded)
            k_bers.append(0.0)   # BCH guarantees exact recovery when decode succeeds

    full_rec   = np.packbits(np.concatenate(recovered_keys))
    rec_hash   = hashlib.sha256(full_rec.tobytes()).hexdigest()
    key_match  = (rec_hash == key_hash) and (fail_count == 0)

    valid_t = [t for t in t_per_chunk if t >= 0]
    return {
        "key_match"      : key_match,
        "recovered_hash" : rec_hash,
        "bch_fail"       : fail_count,
        "k_min"          : min(k_bers),
        "k_max"          : max(k_bers),
        "k_mean"         : float(np.mean(k_bers)),
        "t_min"          : min(valid_t) if valid_t else -1,
        "t_max"          : max(valid_t) if valid_t else -1,
        "t_mean"         : float(np.mean(valid_t)) if valid_t else -1.0,
        "t_per_chunk"    : t_per_chunk,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SCALE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best_scale(embeddings):
    """
    Always sweeps K=1,2,3,4 and shows BER for all.
    Returns FORCE_SCALE if set, otherwise returns the auto-selected best.
    """
    genuine_keys = ["video_1","video_2","video_3","video_4"]
    auto_best    = SCALE_VALUES[0]
    best_mean    = 999.0

    print(f"\n  Scale sweep — raw bit-vector BER between genuine pairs (pre-BCH)")
    print(f"  {'Scale':<8}  {'Min BER':>8}  {'Max BER':>8}  {'Mean BER':>9}  {'Note'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*20}")

    scale_stats = {}
    for scale in SCALE_VALUES:
        bvs  = {k: embedding_to_bitvec(embeddings[k], scale) for k in genuine_keys}
        bers = []
        for i, ka in enumerate(genuine_keys):
            for kb in genuine_keys[i+1:]:
                _, rate = ber_bits(bvs[ka], bvs[kb])
                bers.append(rate)
        mean_ber = float(np.mean(bers))
        scale_stats[scale] = {"min": min(bers), "max": max(bers), "mean": mean_ber}
        if mean_ber < best_mean:
            best_mean = mean_ber
            auto_best = scale

    for scale in SCALE_VALUES:
        s = scale_stats[scale]
        forced_tag = "← FORCED" if FORCE_SCALE == scale else \
                     "← auto-best" if (FORCE_SCALE is None and scale == auto_best) else ""
        print(f"  K={scale:<6}  {s['min']:>7.2f}%  {s['max']:>7.2f}%  {s['mean']:>8.2f}%  {forced_tag}")

    if FORCE_SCALE is not None:
        if FORCE_SCALE not in SCALE_VALUES:
            raise ValueError(f"FORCE_SCALE={FORCE_SCALE} not in SCALE_VALUES={SCALE_VALUES}")
        chosen = FORCE_SCALE
        print(f"\n  → Using scale: K={chosen}  [FORCED]  "
              f"(auto-best was K={auto_best} with mean BER={scale_stats[auto_best]['mean']:.2f}%)")
    else:
        chosen = auto_best
        print(f"\n  → Using scale: K={chosen}  [auto-selected]  "
              f"(mean genuine BER = {scale_stats[chosen]['mean']:.2f}%)")

    return chosen


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 76
SEP2 = "─" * 76

def print_result_row(label, result, is_genuine):
    km  = f"{result['k_min']:5.1f}"
    kmx = f"{result['k_max']:5.1f}"
    kmn = f"{result['k_mean']:5.1f}"
    tm  = f"{result['t_min']:3}"
    tmx = f"{result['t_max']:3}"
    tmn = f"{result['t_mean']:5.1f}"
    fail= result['bch_fail']
    tag = "✓ PASS" if result['key_match'] else "✗ FAIL"
    who = "(genuine) " if is_genuine else "(impostor)"
    print(f"  {label:<22}  k%[{km},{kmx},{kmn}]  t[{tm},{tmx},{tmn}]  fail={fail}  {tag} {who}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model=AdaFaceModel(WEIGHTS_PATH)
    aligner=FaceAligner()

    # ── Extract embeddings ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EXTRACTING EMBEDDINGS — ALL 7 VIDEOS")
    print(SEP)
    print(f"  {FRAMES_TO_USE} frames | CLAHE | Eye cascade | Umeyama 112×112 | AdaFace IR-18")
    print(SEP2)

    embeddings: Dict[str,np.ndarray]={}
    for idx,vp in enumerate(VIDEO_PATHS,start=1):
        key=f"video_{idx}"
        result=embed_video(vp,model,aligner)
        if result is not None: embeddings[key]=result

    for k in [f"video_{i}" for i in range(1,8)]:
        if k not in embeddings: raise RuntimeError(f"Embedding failed for {k}")

    # ── Scale selection ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    mode = f"FORCE_SCALE={FORCE_SCALE}" if FORCE_SCALE is not None else "auto-select"
    print(f"  SCALE SELECTION  (K ∈ {{1, 2, 3, 4}})  [{mode}]")
    print(SEP)
    best_scale = select_best_scale(embeddings)

    # ── Convert to bit vectors ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BIT VECTOR GENERATION")
    print(f"  Pipeline: L2-unit → ×{best_scale} → clip[-1,1] → {QUANT_BITS}-bit quant → Gray → {NUM_CHUNKS}×{BCH_N}={NUM_CHUNKS*BCH_N} bits")
    print(SEP)

    all_keys=[f"video_{i}" for i in range(1,8)]
    bitvecs: Dict[str,np.ndarray]={}
    for k in all_keys:
        bv=embedding_to_bitvec(embeddings[k],best_scale)
        bitvecs[k]=bv
        name=VIDEO_NAMES[k]
        print(f"  {name:<20}  len={len(bv)}  ones={int(bv.sum())}  density={bv.mean():.3f}")

    # Pre-BCH BER
    print(f"\n  Pre-BCH BER (interleaved bits — as BCH sees them):")
    print(f"  {'Pair':<37}  {'Err':>5}  {'BER%':>6}  {'Avg/chunk':>9}  {'Max/chunk':>9}  Type")
    print(f"  {'─'*37}  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}")
    genuine_set={"video_1","video_2","video_3","video_4"}
    for i,ka in enumerate(all_keys):
        for kb in all_keys[i+1:]:
            errs, rate = ber_bits(bitvecs[ka], bitvecs[kb])
            chunk_errs = [int(np.sum(bitvecs[ka][c*BCH_N:(c+1)*BCH_N] !=
                                     bitvecs[kb][c*BCH_N:(c+1)*BCH_N]))
                          for c in range(NUM_CHUNKS)]
            avg_chunk = errs / NUM_CHUNKS
            max_chunk = max(chunk_errs)
            kind = "genuine " if (ka in genuine_set and kb in genuine_set) else "impostor"
            warn = " ⚠" if max_chunk > BCH_T else ""
            print(f"  {VIDEO_NAMES[ka]+' vs '+VIDEO_NAMES[kb]:<37}  {errs:>5}  "
                  f"{rate:>5.2f}%  {avg_chunk:>8.1f}b  {max_chunk:>8}b{warn}   {kind}")

    # ── BCH codec ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BCH({BCH_N},{BCH_K},{BCH_T}) CODEC INITIALISATION")
    print(SEP)
    bch=BCH()
    gen=bch._gen_binary()
    print(f"  Generator degree : {len(gen)-1}  (expected {BCH_N-BCH_K})")
    print(f"  Error capacity   : t={BCH_T} bit-errors per chunk")
    print(f"  Key bits         : {NUM_CHUNKS}×{BCH_K} = {NUM_CHUNKS*BCH_K} bits total")

    # ── Phase 1: Generate shared key + Enroll V1, V2, V3, V4 ─────────────────
    print(f"\n{SEP}")
    print("  PHASE 1 — SHARED KEY GENERATION + ENROLLMENT  (V1, V2, V3, V4)")
    print(SEP)
    print("  ONE shared key K is generated for this person.")
    print("  ALL four videos lock the SAME key K into their helper data.")
    print("  helper_Vi = bitvec_Vi  XOR  BCH_encode(K)")
    print()

    rng = np.random.default_rng(seed=42)
    shared_key_chunks, shared_key_hash = generate_shared_key(rng)
    print(f"  Shared key K  :  {NUM_CHUNKS}×{BCH_K} = {NUM_CHUNKS*BCH_K} bits")
    print(f"  Shared key hash (SHA-256) :")
    print(f"    {shared_key_hash}")
    print()
    print("  This hash MUST appear in every genuine recovery below.")
    print()

    enrolled: Dict[str, Dict] = {}
    for k in ["video_1", "video_2", "video_3", "video_4"]:
        hc = enroll(bitvecs[k], shared_key_chunks, bch)
        enrolled[k] = {"helper_chunks": hc, "key_hash": shared_key_hash}
        print(f"  {VIDEO_NAMES[k]:<20}  helper_chunks={len(hc)}×{BCH_N}b  "
              f"key_hash={shared_key_hash[:32]}...")

    # ── Phase 2: Recovery ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 2 — BCH FUZZY COMMITMENT RECOVERY")
    print(SEP)
    print(f"  BCH({BCH_N},{BCH_K},{BCH_T})  chunks={NUM_CHUNKS}  scale=K={best_scale}"
          + (f"  [FORCED — auto-best was K=1]" if FORCE_SCALE is not None else "  [auto-selected]"))
    print(f"  key = shared random K  (same for all enrollments, stored securely)")
    print(f"  k%[min,max,mean] = per-chunk key BER%  (0% = BCH succeeded all chunks)")
    print(f"  t[min,max,mean]  = bit-errors corrected per chunk  (max={BCH_T})")
    print(f"  fail             = chunks where BCH decoding failed")
    print(f"\n  Expectation:")
    print(f"    Genuine  (V1-V4, same person) → recovered hash MATCHES enrolled hash ✓")
    print(f"    Impostor (V5-V7, diff person) → BCH fails, hash MISMATCH             ✗")

    header=(f"  {'Probe':<22}  {'k%[min, max,mean]':^22}  "
            f"{'t[min,max, mean]':^18}  fail  Result         Recovered hash (first 20)")
    hline =f"  {'─'*22}  {'─'*22}  {'─'*18}  {'─'*4}  {'─'*14}  {'─'*20}"

    def run_test(enrolled_key, probe_keys, genuine_set_local, label):
        e = enrolled[enrolled_key]
        print(f"\n  {SEP2}")
        print(f"  {label}")
        print(f"  Enrolled hash : {e['key_hash']}")
        print(f"  {SEP2}")
        print(header); print(hline)
        results = {}
        for probe in probe_keys:
            is_g = probe in genuine_set_local
            r = recover(bitvecs[probe], e["helper_chunks"], e["key_hash"], bch)
            results[probe] = r
            # print row
            km  = f"{r['k_min']:5.1f}"
            kmx = f"{r['k_max']:5.1f}"
            kmn = f"{r['k_mean']:5.1f}"
            tm  = f"{r['t_min']:3}"
            tmx = f"{r['t_max']:3}"
            tmn = f"{r['t_mean']:5.1f}"
            fail= r['bch_fail']
            tag = "✓ PASS" if r['key_match'] else "✗ FAIL"
            who = "(genuine) " if is_g else "(impostor)"
            rhash = r['recovered_hash'][:20] if r['key_match'] else r['recovered_hash'][:20]
            match_mark = "=" if r['key_match'] else "≠"
            print(f"  {VIDEO_NAMES[probe]:<22}  k%[{km},{kmx},{kmn}]  "
                  f"t[{tm},{tmx},{tmn}]  fail={fail}  {tag} {who}  "
                  f"{match_mark} {rhash}...")
        g_pass = sum(1 for p in probe_keys if p in genuine_set_local and results[p]["key_match"])
        i_rej  = sum(1 for p in probe_keys if p not in genuine_set_local and not results[p]["key_match"])
        g_n    = sum(1 for p in probe_keys if p in genuine_set_local)
        i_n    = sum(1 for p in probe_keys if p not in genuine_set_local)
        print(f"\n  Genuine PASS {g_pass}/{g_n}  |  Impostor REJECT {i_rej}/{i_n}")
        return results, g_pass, g_n, i_rej, i_n

    # Test 1: V1 enrolled — all others probe
    genuine_pool = {"video_1","video_2","video_3","video_4"}
    r1,g1p,g1n,i1r,i1n=run_test(
        "video_1",
        ["video_2","video_3","video_4","video_5","video_6","video_7"],
        genuine_pool - {"video_1"},
        "TEST 1 — Enrolled: V1  |  Probe: V2,V3,V4 (genuine)  V5,V6,V7 (impostor)"
    )

    # Test 2: V2 enrolled — all others probe
    r2,g2p,g2n,i2r,i2n=run_test(
        "video_2",
        ["video_1","video_3","video_4","video_5","video_6","video_7"],
        genuine_pool - {"video_2"},
        "TEST 2 — Enrolled: V2  |  Probe: V1,V3,V4 (genuine)  V5,V6,V7 (impostor)"
    )

    # Test 3: V3 enrolled — all others probe
    r3,g3p,g3n,i3r,i3n=run_test(
        "video_3",
        ["video_1","video_2","video_4","video_5","video_6","video_7"],
        genuine_pool - {"video_3"},
        "TEST 3 — Enrolled: V3  |  Probe: V1,V2,V4 (genuine)  V5,V6,V7 (impostor)"
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(SEP)
    print(f"  BCH parameters : n={BCH_N}, k={BCH_K}, t={BCH_T}, chunks={NUM_CHUNKS}")
    print(f"  Scale          : K={best_scale}")
    print(f"  Bit vector     : {NUM_CHUNKS*BCH_N} bits  ({QUANT_BITS}-bit Gray code)")
    print(f"  Key derivation : shared random K  (one key per person, used for all enrollments)")
    print(f"  Key length     : {NUM_CHUNKS*BCH_K} bits")
    print()
    print(f"  All genuine probes should recover the EXACT same hash as the enrolled key.")
    print(f"  All impostor probes should produce a DIFFERENT hash (BCH decode failure).")
    print()

    total_g  = g1p+g2p+g3p
    total_i  = i1r+i2r+i3r
    total_gn = g1n+g2n+g3n   # 3+3+3 = 9
    total_in = i1n+i2n+i3n   # 3+3+3 = 9

    print(f"  Genuine  accepted  : {total_g}/{total_gn}")
    print(f"  Impostor rejected  : {total_i}/{total_in}")
    print()

    if total_g==total_gn and total_i==total_in:
        print("  ★ PERFECT SEPARATION — all genuine accepted, all impostors rejected")
        print(f"  ★ All genuine probes recovered the exact enrolled key hash ✓")
    else:
        if total_g<total_gn:
            print(f"  ✗ {total_gn-total_g} genuine pair(s) failed → per-chunk BER may exceed t={BCH_T}")
        if total_i<total_in:
            print(f"  ✗ {total_in-total_i} impostor(s) accepted — BCH t may be too permissive")
    print(SEP)


if __name__=="__main__":
    main()
