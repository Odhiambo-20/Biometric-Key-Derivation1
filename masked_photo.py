"""
FACE EMBEDDING PIPELINE WITH COSINE SIMILARITY
5 Videos | 20 HIGH-QUALITY Frames each | 1 Saved masked photo per video

Videos:
  1. IOS.mov
  2. IOS M-No Beard .mov
  3. Android .mp4
  4. Android M-No Beard .mp4
  5. Android video 5.mp4  <- v5

Mask placement:
  - Covers ONLY: below nostrils -> mouth, chin  (bottom ~38%)
  - Keeps visible: forehead, eyebrows, eyes, nose INCLUDING NOSTRILS (top ~62%)

MASK_FRACTION = 0.38:
  black_rows = int(112 * 0.38) = 42 rows  -> rows 70..111 = black (below nostrils)
  visible    = 112 - 42        = 70 rows  -> rows  0..69  = forehead/eyes/nose/nostrils

High-quality frame selection:
  - Candidate pool: 60 evenly spaced frames extracted from the video
  - Quality metric: Laplacian variance (measures sharpness/focus)
  - Final selection: top 20 sharpest frames from the pool
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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


# -- CONFIG ------------------------------------------------------------------

VIDEO_PATHS = [
    "/home/victor/Documents/Desktop/Embeddings/IOS.mov",
    "/home/victor/Documents/Desktop/Embeddings/IOS M-No Beard .mov",
    "/home/victor/Documents/Desktop/Embeddings/Android .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4",
    "/home/victor/Documents/Desktop/Embeddings/Android video 5.mp4",   # <- v5
]

# Human-readable labels aligned by index (used in comparisons)
VIDEO_LABELS = {
    "video_1": "IOS",
    "video_2": "IOS_NoBeard",
    "video_3": "Android",
    "video_4": "Android_NoBeard",
    "video_5": "Android_v5",
}

WEIGHTS_PATH         = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
OUTPUT_ROOT          = "masked_frames"   # one folder -- one .jpg per video saved here
FRAMES_TO_USE        = 20               # final frames used for embedding
CANDIDATE_MULTIPLIER = 3                # candidate pool = FRAMES_TO_USE * this  (60 frames scanned)
FACE_SIZE            = 112              # AdaFace input size

# -- MASK FRACTION -----------------------------------------------------------
# 0.38 -> bottom 38% black = rows 70..111  (mouth + chin ONLY, nostrils SAFE)
#          top   62% shown = rows  0..69   (forehead, eyes, nose, nostrils)
MASK_FRACTION = 0.38

# -- QUALITY THRESHOLDS ------------------------------------------------------
MIN_SIMILARITY_THRESHOLD = 0.80  # Minimum expected similarity for same person


# -- MASK --------------------------------------------------------------------

def apply_mask(image: np.ndarray) -> np.ndarray:
    """
    Black out the bottom MASK_FRACTION of the 112x112 face crop.

    MASK_FRACTION = 0.38:
      black_rows = int(112 * 0.38) = 42
      rows  0..69  -> visible  (forehead, eyebrows, eyes, nose, NOSTRILS all kept)
      rows 70..111 -> black    (mouth, chin hidden -- nostrils NOT touched)
    """
    img        = image.copy()
    h          = img.shape[0]
    black_rows = int(h * MASK_FRACTION)
    img[-black_rows:, :] = 0
    return img


# -- QUALITY METRIC ----------------------------------------------------------

def sharpness_score(gray_frame: np.ndarray) -> float:
    """Laplacian variance -- higher = sharper / better quality."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


# -- FACE DETECTOR -----------------------------------------------------------

class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(xml)
        if self.det.empty():
            raise RuntimeError("Haar cascade XML not found.")
        log.info("Face detector ready.")

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return tightly cropped face BGR array (with small padding), or None."""
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.det.detectMultiScale(
            gray,
            scaleFactor  = 1.05,
            minNeighbors = 6,
            minSize      = (80, 80),
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


# -- ADAFACE MODEL -----------------------------------------------------------

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
        log.info(f"Model loaded | Provider: {providers[0]}")

    def get_embedding(self, face_112: np.ndarray) -> np.ndarray:
        """
        112x112 BGR masked face -> 512-dim L2-normalised float32 embedding.
        AdaFace pre-processing: BGR->RGB, /255, (x-0.5)/0.5, CHW, batch dim.
        """
        img = cv2.resize(face_112, (FACE_SIZE, FACE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 112, 112)

        out = self.session.run([self.output_name], {self.input_name: img})
        emb = out[0][0] if out[0].ndim == 2 else out[0]

        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            raise ValueError("Embedding norm near zero -- bad crop?")
        emb = (emb / norm).astype(np.float32)
        return emb


# -- HIGH-QUALITY FRAME EXTRACTION -------------------------------------------

def extract_high_quality_frames(video_path: str, num_frames: int) -> List[Tuple[int, np.ndarray]]:
    """
    1. Sample a CANDIDATE pool of (num_frames * CANDIDATE_MULTIPLIER) frames
       spread evenly across the video.
    2. Score each candidate frame by Laplacian variance (sharpness).
    3. Return the top `num_frames` sharpest frames as (position, frame) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    n_candidates = num_frames * CANDIDATE_MULTIPLIER
    log.info(f"  {Path(video_path).name} -- {total} frames @ {fps:.1f} fps")
    log.info(f"  Scanning {n_candidates} candidate frames to pick top {num_frames} by sharpness")

    positions = [
        int(round(i * (total - 1) / (n_candidates - 1)))
        for i in range(n_candidates)
    ]

    candidates = []   # (sharpness, pos, frame)
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = sharpness_score(gray)
        candidates.append((score, pos, frame))

    cap.release()

    if not candidates:
        raise RuntimeError("No frames could be read from the video.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:num_frames]
    top.sort(key=lambda x: x[1])

    log.info(f"  Sharpness range (selected): "
             f"{top[0][0]:.1f} .. {top[-1][0]:.1f}  "
             f"(pool max={candidates[0][0]:.1f})")

    frames = [(pos, frame) for _, pos, frame in top]
    log.info(f"  High-quality frames selected: {len(frames)}/{n_candidates} candidates")
    return frames


# -- PRINT EMBEDDING ---------------------------------------------------------

def print_embedding(embedding: np.ndarray, video_name: str):
    """Print all 512 dimensions of the final embedding."""
    label = VIDEO_LABELS.get(video_name, video_name)
    print(f"\n  FINAL EMBEDDING FOR {video_name} ({label}) (512 dimensions):")
    print("  " + "-" * 60)
    for i in range(512):
        value = embedding[i]
        sign  = "+" if value >= 0 else ""
        print(f"  Dim {i+1:3}: {sign}{value:.8f}")
    print("  " + "-" * 60)
    norm = np.linalg.norm(embedding)
    print(f"  Embedding norm: {norm:.8f}")
    print("  " + "-" * 60)


# -- PROCESS ONE VIDEO -------------------------------------------------------

def process_video(
    video_path  : str,
    video_index : int,
    model       : AdaFaceModel,
    detector    : FaceDetector,
) -> Optional[Tuple[str, np.ndarray]]:

    sep        = "-" * 60
    name       = Path(video_path).name
    video_name = f"video_{video_index}"
    label      = VIDEO_LABELS.get(video_name, video_name)
    print(f"\n{sep}\n  VIDEO {video_index} [{label}]: {name}\n{sep}")

    # 1. Extract top-20 sharpest frames
    frames = extract_high_quality_frames(video_path, FRAMES_TO_USE)
    if not frames:
        log.error("No frames extracted.")
        return None

    # 2. Detect faces in each quality frame
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

    # 3. Resize -> mask -> L2-normalise embed each crop
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

    # 4. Save ONE masked photo per video
    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path  = output_dir / f"video_{video_index}_masked.jpg"
    cv2.imwrite(str(save_path), best_masked)
    log.info(f"  Masked photo saved -> {save_path}")

    # 5. Average L2-normalised embeddings -> renormalise
    stack = np.stack(embeddings, axis=0)
    avg   = np.mean(stack, axis=0).astype(np.float32)
    norm  = float(np.linalg.norm(avg))
    if norm < 1e-10:
        raise ValueError("Averaged embedding norm near zero.")
    final = (avg / norm).astype(np.float32)

    # 6. Print summary
    black_rows   = int(FACE_SIZE * MASK_FRACTION)
    visible_rows = FACE_SIZE - black_rows

    print(f"\n  Frames extracted (high-quality) : {len(frames)}")
    print(f"  Frames with detected face       : {len(crops)}")
    print(f"  Mask cut line                   : row {visible_rows} of 112")
    print(f"    Visible rows 0..{visible_rows-1:>2}           : forehead, eyes, nose, NOSTRILS")
    print(f"    Black  rows {visible_rows}..111          : mouth, chin (nostrils untouched)")
    print(f"  Saved photo                     : {save_path}")

    # 7. Print all 512 embedding values
    print_embedding(final, video_name)

    return video_name, final


# -- COSINE SIMILARITY -------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, label: str = "") -> float:
    """Cosine similarity between two L2-normalized embeddings."""
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
        log.info(f"Cosine similarity [{label}] : {sim:.8f}")

    return sim


def compute_pairwise_similarities(embeddings_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute pairwise cosine similarities between all video embeddings."""
    video_names  = list(embeddings_dict.keys())
    similarities = {}

    print("\n" + "=" * 60)
    print("  PAIRWISE COSINE SIMILARITY COMPARISONS")
    print("  (Using FINAL renormalized embeddings from each video)")
    print("=" * 60)

    pairs = list(itertools.combinations(video_names, 2))

    for v1_name, v2_name in pairs:
        emb1   = embeddings_dict[v1_name]
        emb2   = embeddings_dict[v2_name]
        norm1  = np.linalg.norm(emb1)
        norm2  = np.linalg.norm(emb2)
        label1 = VIDEO_LABELS.get(v1_name, v1_name)
        label2 = VIDEO_LABELS.get(v2_name, v2_name)

        print(f"\n  {v1_name} [{label1}] (norm={norm1:.8f})")
        print(f"    vs  {v2_name} [{label2}] (norm={norm2:.8f})")

        sim      = cosine_similarity(emb1, emb2, label=f"{v1_name}_vs_{v2_name}")
        pair_key = f"{v1_name}_vs_{v2_name}"
        similarities[pair_key] = sim

        status = "GOOD" if sim >= MIN_SIMILARITY_THRESHOLD else "LOW"
        print(f"  {'='*40}")
        print(f"  COSINE SIMILARITY: {sim:.8f}   {status}")
        print(f"  {'='*40}")

        if sim < MIN_SIMILARITY_THRESHOLD:
            log.warning(f"Similarity {sim:.4f} below expected threshold {MIN_SIMILARITY_THRESHOLD}")

    return similarities


def analyze_similarities(similarities: Dict[str, float]) -> Dict:
    """Analyze similarity results and generate statistics."""
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
        'all_similarities'       : all_sims,
        'min_similarity'         : min(all_sims),
        'max_similarity'         : max(all_sims),
        'mean_similarity'        : float(np.mean(all_sims)),
        'std_dev'                : float(np.std(all_sims)),
        'most_similar_pair'      : max_pair,
        'least_similar_pair'     : min_pair,
        'low_similarities_count' : len(low_sims),
        'low_similarities_pairs' : low_sims,
        'quality_assessment'     : quality,
        'total_pairs'            : len(all_sims),
    }


def print_similarity_analysis(analysis: Dict):
    """Print the similarity analysis results."""
    if not analysis:
        return

    print("\n" + "-" * 60)
    print("  SIMILARITY RANGE")
    print("-" * 60)

    mp   = analysis['most_similar_pair']
    lp   = analysis['least_similar_pair']
    ml1, ml2 = mp[0].split("_vs_")
    ll1, ll2 = lp[0].split("_vs_")

    print(f"  Most similar  : {ml1} [{VIDEO_LABELS.get(ml1, ml1)}]"
          f" <-> {ml2} [{VIDEO_LABELS.get(ml2, ml2)}]  =  {mp[1]:.8f}")
    print(f"  Least similar : {ll1} [{VIDEO_LABELS.get(ll1, ll1)}]"
          f" <-> {ll2} [{VIDEO_LABELS.get(ll2, ll2)}]  =  {lp[1]:.8f}")
    print("-" * 60)

    print("\n  SIMILARITY STATISTICS")
    print("-" * 60)
    print(f"  Minimum : {analysis['min_similarity']:.8f}")
    print(f"  Maximum : {analysis['max_similarity']:.8f}")
    print(f"  Mean    : {analysis['mean_similarity']:.8f}")
    print(f"  Std Dev : {analysis['std_dev']:.8f}")

    if analysis['low_similarities_count'] > 0:
        print(f"\n  NOTE: {analysis['low_similarities_count']}/{analysis['total_pairs']} "
              f"similarities below threshold {MIN_SIMILARITY_THRESHOLD}")
        print("\n  Pairs below threshold:")
        for pair, sim in analysis['low_similarities_pairs']:
            p1, p2 = pair.split("_vs_")
            print(f"     - {p1} [{VIDEO_LABELS.get(p1, p1)}]"
                  f" <-> {p2} [{VIDEO_LABELS.get(p2, p2)}]: {sim:.8f}")
        print("\n  Possible reasons for low similarity:")
        print("     - Different person in some videos")
        print("     - Poor quality frames in one video")
        print("     - Extreme pose/lighting variations")
        print("     - Face detection/alignment issues")
    else:
        print(f"\n  All {analysis['total_pairs']} pairs above threshold {MIN_SIMILARITY_THRESHOLD}")
    print("-" * 60)


# -- v5-SPECIFIC COMPARISON SECTION ------------------------------------------

def print_v5_comparisons(similarities: Dict[str, float]):
    """Print a focused summary of how v5 (video_5) compares to all others."""
    v5_pairs = {k: v for k, v in similarities.items() if "video_5" in k}
    if not v5_pairs:
        return

    print("\n" + "=" * 60)
    print("  VIDEO 5 (Android_v5) -- COMPARISONS SUMMARY")
    print("=" * 60)
    for pair, sim in sorted(v5_pairs.items(), key=lambda x: x[1], reverse=True):
        p1, p2 = pair.split("_vs_")
        other  = p2 if p1 == "video_5" else p1
        lbl    = VIDEO_LABELS.get(other, other)
        status = "GOOD" if sim >= MIN_SIMILARITY_THRESHOLD else "LOW"
        print(f"  video_5 <-> {other:8} [{lbl:20}]:  {sim:.8f}  {status}")
    print("=" * 60)


# -- MAIN --------------------------------------------------------------------

def run():
    sep          = "=" * 60
    black_rows   = int(FACE_SIZE * MASK_FRACTION)
    visible_rows = FACE_SIZE - black_rows

    print(sep)
    print("  ADAFACE HIGH-QUALITY FRAME PIPELINE  (5 videos)")
    print(sep)
    for i, vp in enumerate(VIDEO_PATHS, 1):
        label = VIDEO_LABELS.get(f"video_{i}", f"video_{i}")
        print(f"    {i}. [{label}]  {Path(vp).name}")
    print(f"\n  Frame strategy    : top {FRAMES_TO_USE} sharpest from "
          f"{FRAMES_TO_USE * CANDIDATE_MULTIPLIER} candidates")
    print(f"  Quality metric    : Laplacian variance (sharpness)")
    print(f"  Resize method     : INTER_LANCZOS4 (highest quality)")
    print(f"  Mask fraction     : {MASK_FRACTION} -> {black_rows} rows black "
          f"(rows {visible_rows}..111)")
    print(f"  Visible rows 0..{visible_rows-1} : forehead + eyebrows + eyes + nose + NOSTRILS")
    print(f"  Black rows {visible_rows}..111   : mouth + chin ONLY")
    print(f"  Photos saved to   : {OUTPUT_ROOT}/")
    print(sep)

    # Validate paths
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    for vp in VIDEO_PATHS:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    # Process all 5 videos
    results: Dict[str, np.ndarray] = {}
    for idx, vp in enumerate(VIDEO_PATHS, start=1):
        result = process_video(vp, idx, model, detector)
        if result is not None:
            video_name, final_emb = result
            results[video_name] = final_emb

    # Summary of processed videos
    print(f"\n{sep}")
    print("  PROCESSING COMPLETE -- FINAL EMBEDDINGS SUMMARY")
    print(sep)
    for name, emb in results.items():
        label = VIDEO_LABELS.get(name, name)
        norm  = np.linalg.norm(emb)
        print(f"  {name:10} [{label:20}]: norm={norm:.8f}")
    print(f"\n  Photos saved to: {Path(OUTPUT_ROOT).resolve()}/")
    print(sep)

    # FIX: initialise before the conditional block to avoid NameError
    similarities: Dict[str, float] = {}

    if len(results) >= 2:
        similarities = compute_pairwise_similarities(results)

        if similarities:
            analysis = analyze_similarities(similarities)
            print_similarity_analysis(analysis)

            # Focused v5 summary
            print_v5_comparisons(similarities)

            avg_sim = analysis['mean_similarity']
            print("\n" + "-" * 60)
            print(f"  AVERAGE SIMILARITY (all {analysis['total_pairs']} pairs) : {avg_sim:.8f}")
            print(f"  OVERALL QUALITY: {analysis['quality_assessment']}")
            print("-" * 60)
    else:
        print(f"\n{sep}")
        print("  NOTE: Need at least 2 videos for pairwise comparison")
        print(f"  Only {len(results)} video(s) processed successfully")
        print(sep)

    return results, similarities


# -- ENTRY POINT -------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" STARTING ADAFACE EMBEDDING PIPELINE WITH COSINE SIMILARITY  (5 videos)")
    print("=" * 80)

    embeddings, pairwise_sims = run()

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETE -- FINAL RESULTS")
    print("=" * 80)
    print(f"  Videos processed successfully : {len(embeddings)}/{len(VIDEO_PATHS)}")
    print(f"  Pairs compared                : {len(pairwise_sims)}")

    if pairwise_sims:
        all_sims = list(pairwise_sims.values())
        avg_sim  = float(np.mean(all_sims))
        print(f"  Average similarity            : {avg_sim:.8f}")
        print(f"  Similarity range              : {min(all_sims):.8f} - {max(all_sims):.8f}")

        low_sims = [s for s in all_sims if s < MIN_SIMILARITY_THRESHOLD]
        if len(low_sims) == 0:
            assessment = "GOOD -- All similarities within expected range"
        elif len(low_sims) <= len(all_sims) / 3:
            assessment = "FAIR -- Some similarities below threshold"
        else:
            assessment = "POOR -- Multiple similarities below threshold"

        print(f"\n  OVERALL ASSESSMENT: {assessment}")
        if low_sims:
            print(f"     {len(low_sims)} out of {len(all_sims)} pairs below threshold "
                  f"{MIN_SIMILARITY_THRESHOLD}")
            print(f"     Lowest similarity: {min(low_sims):.8f}")

    print("=" * 80)





