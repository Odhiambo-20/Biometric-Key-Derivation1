"""
FACE EMBEDDING PIPELINE - Single Video, 10 Frames, L2 Renormalization

Input    : One video file
Model    : AdaFace ONNX  (adaface_ir_18.onnx)

Pipeline:
  Phase 1  - Extract 10 evenly spaced frames from the video
  Phase 2  - Detect and crop face from each frame (Haar cascade)
  Phase 3  - Extract 512-dim AdaFace embedding from each face crop
  Phase 4  - Average all 10 embeddings into one vector
  Phase 5  - L2 renormalize the averaged vector (norm must equal 1.0)

SETUP:
  pip install onnxruntime opencv-python numpy --break-system-packages

RUN:
  python3 embedding_pipeline.py
"""

import logging
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


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

VIDEO_PATH     = "/home/victor/Documents/Desktop/Embeddings/Android M-No Beard .mp4"
WEIGHTS_PATH   = "/home/victor/Documents/Desktop/Adaface/adaface-onnx/weights/adaface_ir_18.onnx"
FRAMES_TO_USE  = 10
FACE_PADDING   = 0.2


# ------------------------------------------------------------------
# ADAFACE MODEL
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# FACE DETECTOR
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# PHASE 1 - EXTRACT 10 EVENLY SPACED FRAMES
# ------------------------------------------------------------------

def extract_frames(video_path: str, num_frames: int) -> list:
    """
    Open the video and extract num_frames evenly spaced across its duration.

    For a 323-frame video with num_frames=10:
      positions = [0, 35, 71, 107, 143, 179, 215, 251, 287, 322]

    Returns
    -------
    list of np.ndarray  BGR frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    log.info(f"Video : {Path(video_path).name}")
    log.info(f"Total frames : {total_frames}    FPS : {fps}    Duration : {duration:.2f}s")

    # Calculate evenly spaced positions
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


# ------------------------------------------------------------------
# PHASE 2 - DETECT FACE IN EACH FRAME
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# PHASE 3 - EXTRACT EMBEDDING FROM EACH FACE CROP
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# PHASE 4 - AVERAGE ALL 10 EMBEDDINGS
# ------------------------------------------------------------------

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

    avg = np.mean(all_embs, axis=0)
    norm_before = float(np.linalg.norm(avg))

    log.info(f"Averaged embedding shape : {avg.shape}")
    log.info(f"Norm before renorm       : {norm_before:.8f}  (less than 1.0 after averaging)")

    return avg


# ------------------------------------------------------------------
# PHASE 5 - L2 RENORMALIZATION
# ------------------------------------------------------------------

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

    renormalized    = vector / norm
    norm_after      = float(np.linalg.norm(renormalized))

    log.info(f"Norm before renorm : {norm:.8f}")
    log.info(f"Norm after renorm  : {norm_after:.8f}  (should be 1.0)")

    return renormalized


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def run():
    sep = "-" * 55

    print(sep)
    print("  ADAFACE PIPELINE")
    print("  Single video, 10 frames, L2 renormalization")
    print(sep)
    print(f"  Video  : {VIDEO_PATH}")
    print(f"  Model  : {WEIGHTS_PATH}")
    print(f"  Frames : {FRAMES_TO_USE}")
    print(sep)

    # Load model
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")

    model    = AdaFaceModel(WEIGHTS_PATH)
    detector = FaceDetector()

    print()
    print("  PHASE 1 - Extract 10 frames")
    print(sep)
    frames = extract_frames(VIDEO_PATH, FRAMES_TO_USE)

    print()
    print("  PHASE 2 - Detect faces")
    print(sep)
    crops = detect_faces(frames, detector)

    if len(crops) == 0:
        raise RuntimeError("No faces detected in any frame.")

    print()
    print("  PHASE 3 - Extract embeddings")
    print(sep)
    embeddings = extract_embeddings(crops, model)

    print()
    print("  PHASE 4 - Average embeddings")
    print(sep)
    averaged = average_embeddings(embeddings)

    print()
    print("  PHASE 5 - L2 Renormalization")
    print(sep)
    final_embedding = l2_renormalize(averaged)

    # Final summary
    print()
    print(sep)
    print("  FINAL RESULT")
    print(sep)
    print(f"  Frames extracted          : {len(frames)}")
    print(f"  Frames with face          : {len(crops)}")
    print(f"  Embeddings averaged       : {len(embeddings)}")
    print(f"  Final embedding shape     : {final_embedding.shape}")
    print(f"  Final embedding norm      : {float(np.linalg.norm(final_embedding)):.8f}")
    print(f"  Final embedding [0:5]     : {[round(float(x), 6) for x in final_embedding[:5]]}")
    print(f"  Final embedding [0:10]    : {[round(float(x), 6) for x in final_embedding[:10]]}")
    print(sep)

    # Print all 512 values
    print()
    print("  ALL 512 EMBEDDING VALUES")
    print(sep)
    for i, val in enumerate(final_embedding):
        print(f"  Dim {i + 1}: {val:.8f}")
    print(sep)

    return final_embedding


if __name__ == "__main__":
    run()
