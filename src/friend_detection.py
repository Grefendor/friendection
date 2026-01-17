import os
import sys
import warnings
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Suppress FutureWarning from insightface/skimage before importing
warnings.filterwarnings("ignore", category=FutureWarning)

# Keep InsightFace assets inside the project models directory
root_dir = Path(__file__).resolve().parents[1]
models_dir = root_dir / "models"
os.environ.setdefault("INSIGHTFACE_HOME", str(models_dir))

# Import InsightFace (suppress its verbose output)
from insightface.app import FaceAnalysis

class DoorFaceRecognizer:
    def __init__(self, providers=None, det_size=(640, 640), verbose=False):
        # Suppress InsightFace verbose model loading messages
        if verbose:
            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=['detection', 'recognition'],
                providers=providers or ["CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=det_size)
        else:
            # Redirect stdout/stderr during initialization
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=['detection', 'recognition'],
                    providers=providers or ["CPUExecutionProvider"]
                )
                self.app.prepare(ctx_id=0, det_size=det_size)
        self.gallery: Dict[str, np.ndarray] = {}
        self._gallery_names: List[str] = []
        self._gallery_protos: Optional[np.ndarray] = None

    def _embed(self, img: np.ndarray, strict: bool = True) -> Optional[np.ndarray]:
        """
        Extract face embedding from image using detection.
        """
        faces = self.app.get(img)
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        
        min_score = 0.4 if strict else 0.25
        min_size = 112 if strict else 50
        max_angle = 35 if strict else 60
        
        if f.det_score < min_score:
            return None
        w = f.bbox[2] - f.bbox[0]
        h = f.bbox[3] - f.bbox[1]
        if min(w, h) < min_size:
            return None
        
        # Handle pose check - pose may be None when landmark models are skipped
        pose = getattr(f, "pose", None)
        if pose is not None:
            yaw, pitch, _ = pose
            if abs(yaw) > max_angle or abs(pitch) > max_angle:
                return None
        
        return f.normed_embedding

    def _embed_cropped(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from an already-cropped face image.
        Resizes to 112x112 and runs recognition model directly.
        """
        if img is None or img.size == 0:
            return None
        
        # Try detection first (in case image has some margin)
        faces = self.app.get(img)
        if faces:
            f = max(faces, key=lambda x: x.det_score)
            return f.normed_embedding
        
        # If detection fails, use recognition model directly on resized crop
        # Find the recognition model (w600k_r50)
        rec_model = None
        for name, model in self.app.models.items():
            if 'recognition' in name or 'w600k' in str(getattr(model, 'onnx_file', '')):
                rec_model = model
                break
        
        if rec_model is None:
            return None
        
        # Resize to ArcFace input size (112x112)
        face_img = cv2.resize(img, (112, 112))
        
        # Prepare input: BGR -> RGB, HWC -> CHW, normalize
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
        face_input = (face_input - 127.5) / 127.5
        face_input = np.expand_dims(face_input, axis=0)
        
        # Run inference
        try:
            embedding = rec_model.session.run(None, {rec_model.session.get_inputs()[0].name: face_input})[0][0]
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception:
            return None

    def build_gallery(self, db_path: str) -> None:
        """
        Build gallery from friends_db/Name/*.jpg structure.
        Handles both full images and pre-cropped face images.
        """
        gallery = {}
        db = Path(db_path)
        print(f"Building gallery from: {db.resolve()}")
        
        for person_dir in db.iterdir():
            if not person_dir.is_dir():
                continue
            embs = []
            img_files = list(person_dir.glob("*.*"))
            print(f"  {person_dir.name}: found {len(img_files)} files")
            
            for img_p in img_files:
                if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue
                img = cv2.imread(str(img_p))
                if img is None:
                    print(f"    {img_p.name}: failed to read")
                    continue
                
                # Try standard detection first
                e = self._embed(img, strict=False)
                if e is not None:
                    embs.append(e)
                    print(f"    {img_p.name}: OK (detected)")
                else:
                    # Fall back to cropped-face embedding
                    e = self._embed_cropped(img)
                    if e is not None:
                        embs.append(e)
                        print(f"    {img_p.name}: OK (cropped)")
                    else:
                        print(f"    {img_p.name}: no valid face detected")
                    
            if embs:
                proto = np.mean(np.vstack(embs), axis=0)
                proto = proto / np.linalg.norm(proto)
                gallery[person_dir.name] = proto
                print(f"  {person_dir.name}: built prototype from {len(embs)} embeddings")
            else:
                print(f"  {person_dir.name}: no embeddings")
                
        self.gallery = gallery
        # Cache matrix for fast similarity (cosine == dot for L2-normalized vectors)
        if gallery:
            self._gallery_names = list(gallery.keys())
            self._gallery_protos = np.vstack([gallery[n] for n in self._gallery_names])
        else:
            self._gallery_names = []
            self._gallery_protos = None
        print(f"Gallery complete: {list(gallery.keys())}")

    def _get_gallery_matrix(self) -> Tuple[List[str], Optional[np.ndarray]]:
        if not self._gallery_names or self._gallery_protos is None:
            return [], None
        return self._gallery_names, self._gallery_protos

    def identify_all(self, crops: List[np.ndarray], sim_thresh=0.70) -> List[Tuple[Optional[str], float]]:
        """
        Identify all faces in crops. Returns a list of (name, similarity) for each crop.
        Returns (None, sim) if no match above threshold.
        """
        names, protos = self._get_gallery_matrix()
        if protos is None:
            return [(None, 0.0) for _ in crops]
        results = []
        
        for i, crop in enumerate(crops):
            # Try detection first, fall back to cropped embedding
            e = self._embed(crop, strict=True)
            method = "detected"
            if e is None:
                e = self._embed_cropped(crop)
                method = "cropped"
            if e is None:
                print(f"  Crop {i}: no embedding extracted")
                results.append((None, 0.0))
                continue
            
            sims = protos @ e
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            best_name = names[idx] if best_sim >= sim_thresh else None
            print(f"  Crop {i} ({method}): best match {names[idx]} sim={best_sim:.3f}")
            results.append((best_name, best_sim))
        
        return results

    def identify_from_faces(self, faces: List, sim_thresh=0.70) -> List[Tuple[Optional[str], float]]:
        """
        Identify faces using pre-extracted embeddings from InsightFace detection.
        This is more efficient than identify_all() because embeddings are already computed.
        
        Args:
            faces: List of InsightFace Face objects (from app.get())
            sim_thresh: Similarity threshold for positive identification
            
        Returns:
            List of (name, similarity) tuples for each face
        """
        names, protos = self._get_gallery_matrix()
        if protos is None:
            return [(None, 0.0) for _ in faces]
        results = []
        
        for i, face in enumerate(faces):
            # Use the embedding already extracted during detection
            e = getattr(face, 'normed_embedding', None)
            if e is None:
                print(f"  Face {i}: no embedding available")
                results.append((None, 0.0))
                continue
            
            sims = protos @ e
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            best_name = names[idx] if best_sim >= sim_thresh else None
            print(f"  Face {i}: best match {names[idx]} sim={best_sim:.3f}")
            results.append((best_name, best_sim))
        
        return results