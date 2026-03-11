"""
AdaFace Cosine Similarity — All Videos vs All Videos
=====================================================
12 videos (V8–V13, makeup + no-makeup) compared against each other.

Pipeline per video (top-20 sharpest frames):
  1. Extract 60 candidate frames → keep top-20 by Laplacian sharpness
  2. MTCNN detect + 5 landmarks → Umeyama → 112×112 aligned crop
     (Haar cascade fallback if MTCNN misses)
  3. Normalisation on aligned crop:
       a. Gamma correction → target luminance 130  (fixes lighting mismatch)
       b. LAB CLAHE on L channel                   (local contrast equalisation)
       c. A/B channel mean-shift to neutral         (removes foundation colour cast)
  4. AdaFace IR-18: embed(norm) + embed(H-flip) → L2(sum) = frame vector
  5. Average 20 frame vectors → L2-renorm → final unit vector
  6. Cosine similarity = dot(unit_vec_A, unit_vec_B)

Output:
  - Within-identity pairs (same person, makeup vs no-makeup)
  - Full 12×12 cosine similarity matrix (all vs all)
  - Cross-identity pairs sorted by similarity
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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

WEIGHTS_PATH = (
    "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
)

VIDEOS = [
    ("V8",  "makeup",    "/home/victor/Documents/Desktop/Embeddings/V8 instagram Make up.mp4"),
    ("V8",  "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V8 instagram No make up.mp4"),
    ("V9",  "makeup",    "/home/victor/Documents/Desktop/Embeddings/V9 Instagram Make up.mp4"),
    ("V9",  "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V9 Instagram No make up.mp4"),
    ("V10", "makeup",    "/home/victor/Documents/Desktop/Embeddings/V10 Instagram Make up.mp4"),
    ("V10", "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V10 Instagram No make up.mp4"),
    ("V11", "makeup",    "/home/victor/Documents/Desktop/Embeddings/V11 instagram Make up.mp4"),
    ("V11", "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V11 instagram No make up.mp4"),
    ("V12", "makeup",    "/home/victor/Documents/Desktop/Embeddings/V12 instagram Make up.mp4"),
    ("V12", "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V12 instagram No make up.mp4"),
    ("V13", "makeup",    "/home/victor/Documents/Desktop/Embeddings/V13 instagram Make up.mp4"),
    ("V13", "no_makeup", "/home/victor/Documents/Desktop/Embeddings/V13 instagram No make up.mp4"),
]

FRAMES_TO_USE  = 20
CANDIDATE_MULT = 3
FACE_SIZE      = 112
MIN_FACE_PX    = 40

REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

TARGET_LUMINANCE = 130.0
LAB_CLAHE        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
NEUTRAL_A        = 128.0
NEUTRAL_B        = 128.0
THRESHOLD        = 0.75


# =============================================================================
#  NORMALISATION
# =============================================================================

def normalise_face(face_bgr: np.ndarray) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    my, mx = h // 5, w // 5
    grey   = cv2.cvtColor(face_bgr[cy-my:cy+my, cx-mx:cx+mx], cv2.COLOR_BGR2GRAY)
    mean_l = float(grey.mean())
    if mean_l > 5.0:
        gamma    = float(np.clip(np.log(TARGET_LUMINANCE/255.0) / np.log(mean_l/255.0), 0.3, 3.0))
        lut      = np.array([min(255, int((i/255.0)**gamma*255)) for i in range(256)], dtype=np.uint8)
        face_bgr = lut[face_bgr]
    lab     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq    = LAB_CLAHE.apply(l)
    a_eq    = np.clip(a.astype(np.int16) + int(NEUTRAL_A - float(a.mean())), 0, 255).astype(np.uint8)
    b_eq    = np.clip(b.astype(np.int16) + int(NEUTRAL_B - float(b.mean())), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_eq, a_eq, b_eq]), cv2.COLOR_LAB2BGR)


# =============================================================================
#  MTCNN
# =============================================================================

def _load_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn  = MTCNN(keep_all=False, min_face_size=MIN_FACE_PX,
                       thresholds=[0.6, 0.7, 0.7], device=device,
                       post_process=False, select_largest=True)
        log.info(f"MTCNN loaded | device={device}")
        return mtcnn
    except ImportError:
        log.warning("facenet-pytorch not installed — using Haar fallback.")
        return None


# =============================================================================
#  FACE ALIGNER
# =============================================================================

class FaceAligner:
    def __init__(self, mtcnn=None):
        self.mtcnn = mtcnn
        cc = cv2.data.haarcascades
        self.face_cc = cv2.CascadeClassifier(cc + "haarcascade_frontalface_alt2.xml")
        if self.face_cc.empty():
            self.face_cc = cv2.CascadeClassifier(cc + "haarcascade_frontalface_default.xml")
        self.leye_cc = cv2.CascadeClassifier(cc + "haarcascade_lefteye_2splits.xml")
        self.reye_cc = cv2.CascadeClassifier(cc + "haarcascade_righteye_2splits.xml")
        self.clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        log.info(f"FaceAligner | {'MTCNN + Haar fallback' if mtcnn else 'Haar only'}")

    @staticmethod
    def _umeyama(src, dst):
        n = src.shape[0]; mu_s = src.mean(0); mu_d = dst.mean(0)
        sc = src-mu_s; dc = dst-mu_d; vs = (sc**2).sum()/n
        if vs < 1e-10: return None
        U, S, Vt = np.linalg.svd((dc.T@sc)/n)
        d = np.ones(2)
        if np.linalg.det(U)*np.linalg.det(Vt) < 0: d[-1] = -1
        R = U@np.diag(d)@Vt; c = (S*d).sum()/vs; t = mu_d-c*R@mu_s
        M = np.zeros((2,3), dtype=np.float32); M[:,:2]=c*R; M[:,2]=t
        return M

    def _warp(self, frame, lms):
        M = self._umeyama(lms, REFERENCE_PTS)
        if M is None: return None
        return cv2.warpAffine(frame, M, (FACE_SIZE, FACE_SIZE),
                              flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    def _mtcnn_align(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            _, _, lms = self.mtcnn.detect(rgb, landmarks=True)
            if lms is not None and len(lms) > 0:
                return self._warp(frame, lms[0].astype(np.float32))
        except Exception:
            pass
        return None

    def _haar_align(self, frame):
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        best = None
        for sf, mn in [(1.05,5),(1.05,3),(1.10,2)]:
            faces = self.face_cc.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn,
                                                  minSize=(MIN_FACE_PX, MIN_FACE_PX))
            if len(faces) > 0:
                best = tuple(max(faces, key=lambda f: f[2]*f[3])); break
        if best is None: return None
        x, y, w, h = best
        roi = gray[y:y+int(h*0.60), x:x+w]; half = w//2
        def _eye(cc, sub, xo, yo):
            for mn in [5,3,2]:
                d = cc.detectMultiScale(sub, scaleFactor=1.10, minNeighbors=mn,
                                        minSize=(int(w*0.10), int(w*0.10)))
                if len(d) > 0:
                    ex,ey,ew,eh = max(d, key=lambda d: d[2]*d[3])
                    return np.array([x+xo+ex+ew//2, y+yo+ey+eh//2], dtype=np.float32)
            return None
        er = _eye(self.reye_cc, roi[:,:half], 0,    0) or np.array([x+0.30*w, y+0.36*h], dtype=np.float32)
        el = _eye(self.leye_cc, roi[:,half:], half, 0) or np.array([x+0.70*w, y+0.36*h], dtype=np.float32)
        if er[0] > el[0]: er, el = el, er
        lms = np.array([er, el, [x+0.50*w,y+0.60*h], [x+0.35*w,y+0.76*h], [x+0.65*w,y+0.76*h]], dtype=np.float32)
        return self._warp(frame, lms)

    def align(self, frame):
        if self.mtcnn:
            r = self._mtcnn_align(frame)
            if r is not None: return r
        return self._haar_align(frame)


# =============================================================================
#  ADAFACE MODEL
# =============================================================================

class AdaFaceModel:
    def __init__(self, model_path):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider","CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session     = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"AdaFace IR-18 | {providers[0]}")

    def raw_embedding(self, face_bgr):
        img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32)/255.0 - 0.5)/0.5
        img = img.transpose(2,0,1)[np.newaxis]
        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim==2 else out[0]
        return emb.astype(np.float32)


# =============================================================================
#  FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan = FRAMES_TO_USE * CANDIDATE_MULT
    positions = [int(round(i*(total-1)/max(n_scan-1,1))) for i in range(n_scan)]
    candidates = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        candidates.append((score, pos, frame))
    cap.release()
    if not candidates: raise RuntimeError(f"No frames: {video_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:FRAMES_TO_USE]
    top.sort(key=lambda x: x[1])
    return [f for _,_,f in top]


def _l2(v):
    n = np.linalg.norm(v)
    return (v/n).astype(np.float32) if n > 1e-10 else v.astype(np.float32)


# =============================================================================
#  EMBED VIDEO
# =============================================================================

def embed_video(video_path, model, aligner, label) -> Optional[np.ndarray]:
    frames    = extract_frames(video_path)
    unit_vecs = []
    for frame in frames:
        aligned = aligner.align(frame)
        if aligned is None: continue
        norm      = normalise_face(aligned)
        e_orig    = model.raw_embedding(norm)
        e_flip    = model.raw_embedding(cv2.flip(norm, 1))
        unit_vecs.append(_l2(e_orig + e_flip))
    if not unit_vecs:
        log.error(f"No faces: {Path(video_path).name}"); return None
    final = _l2(np.mean(np.stack(unit_vecs), axis=0).astype(np.float32))
    log.info(f"  {label:<30}  faces={len(unit_vecs):>2}/{len(frames)}  norm={np.linalg.norm(final):.6f}")
    return final


# =============================================================================
#  MAIN
# =============================================================================

SEP  = "=" * 72
SEP2 = "-" * 72

def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    for _,_,vp in VIDEOS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    mtcnn   = _load_mtcnn()
    model   = AdaFaceModel(WEIGHTS_PATH)
    aligner = FaceAligner(mtcnn=mtcnn)

    # ── Embed all 12 videos ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  EXTRACTING EMBEDDINGS  (20 frames per video)")
    print(SEP)
    print("  Detector      : MTCNN + Haar fallback")
    print("  Normalisation : gamma → LAB CLAHE → A/B neutral shift")
    print("  Per frame     : embed(norm) + embed(H-flip) → L2-norm")
    print("  Final         : mean of 20 frame vectors → L2-renorm → unit vector")
    print(SEP2)

    embeddings: Dict[str, np.ndarray] = {}
    labels: List[str] = []

    for person, variant, vpath in VIDEOS:
        key   = f"{person}_{variant}"
        label = f"{person} {'MU' if variant == 'makeup' else 'NM'}"
        emb   = embed_video(vpath, model, aligner, f"{person} {variant.replace('_',' ')}")
        if emb is None:
            raise RuntimeError(f"Embedding failed for {key}")
        embeddings[key] = emb
        labels.append(key)

    n = len(labels)

    # Short display labels
    def short(key):
        p, v = key.rsplit("_", 1)
        return f"{p}-{'MU' if v == 'makeup' else 'NM'}"

    slabels = [short(k) for k in labels]

    # ── Build full similarity matrix ───────────────────────────────────────
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = float(np.dot(embeddings[labels[i]], embeddings[labels[j]]))

    # ── SECTION 1: Within-identity pairs ──────────────────────────────────
    print(f"\n{SEP}")
    print("  WITHIN-IDENTITY  —  same person: makeup vs no-makeup")
    print(SEP)
    print(f"  {'Pair':<42}  {'Cosine Sim':>10}")
    print(f"  {'-'*42}  {'-'*10}")

    persons      = ["V8", "V9", "V10", "V11", "V12", "V13"]
    within_scores = []
    for person in persons:
        mu_emb = embeddings[f"{person}_makeup"]
        nm_emb = embeddings[f"{person}_no_makeup"]
        s      = float(np.dot(mu_emb, nm_emb))
        within_scores.append(s)
        print(f"  {person+' makeup':<20}  vs  {person+' no makeup':<18}  {s:>10.4f}")

    print()
    print(f"  Mean   : {np.mean(within_scores):.4f}")
    print(f"  Min    : {np.min(within_scores):.4f}  ({persons[int(np.argmin(within_scores))]})")
    print(f"  Max    : {np.max(within_scores):.4f}  ({persons[int(np.argmax(within_scores))]})")
    passing = sum(1 for s in within_scores if s >= THRESHOLD)
    print(f"  Passing: {passing}/{len(within_scores)} above {THRESHOLD}")

    # ── SECTION 2: Full 12×12 matrix ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  FULL 12×12 COSINE SIMILARITY MATRIX  (all videos vs all videos)")
    print(SEP)
    print("  MU = Make up   NM = No make up   Diagonal = 1.000 (self)")
    print()

    col_w  = 8
    header = " " * 10 + "".join(f"{s:>{col_w}}" for s in slabels)
    print(f"  {header}")
    print(f"  {'-'*(10 + col_w*n)}")
    for i, row_lbl in enumerate(slabels):
        row = f"  {row_lbl:<8}"
        for j in range(n):
            val = sim[i, j]
            row += f"{val:>{col_w}.4f}"
        print(row)

    # ── SECTION 3: All unique cross-video pairs sorted ─────────────────────
    print(f"\n{SEP}")
    print("  ALL PAIRWISE COSINE SIMILARITIES  (upper triangle, sorted high → low)")
    print(SEP)
    print(f"  {'Pair':<45}  {'Cosine Sim':>10}  {'Type'}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*16}")

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pi = labels[i].split("_")[0]
            pj = labels[j].split("_")[0]
            same_person = (pi == pj)
            pair_type   = "same person" if same_person else "different person"
            pairs.append((slabels[i], slabels[j], sim[i,j], pair_type))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for la, lb, s, pt in pairs:
        print(f"  {la:<20}  vs  {lb:<22}  {s:>10.4f}  {pt}")

    # ── SECTION 4: Summary ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    cross_scores = [s for _,_,s,pt in pairs if pt == "different person"]
    print(f"  Within-identity mean  : {np.mean(within_scores):.4f}")
    print(f"  Cross-identity  mean  : {np.mean(cross_scores):.4f}")
    gap = np.mean(within_scores) - np.mean(cross_scores)
    print(f"  Separation gap        : {gap:.4f}  "
          f"({'Good' if gap > 0.15 else 'Low'} separation)")
    print(f"  Threshold             : {THRESHOLD}")
    print(f"  Within-identity pass  : {passing}/{len(within_scores)}")
    print()
    print("  Per-person within-identity similarity:")
    for person, s in zip(persons, within_scores):
        print(f"    {person}  {s:.4f}")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
