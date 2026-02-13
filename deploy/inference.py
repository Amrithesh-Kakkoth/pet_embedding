"""Inference pipeline for pet face recognition."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import PetFaceEmbedder


class PetRecognitionPipeline:
    """Complete pipeline: detect -> embed -> match."""

    def __init__(
        self,
        detector_path: str | Path,
        embedder_path: str | Path,
        gallery_path: str | Path | None = None,
        image_size: int = 160,
        device: str = "auto",
    ):
        self.image_size = image_size
        self.device = self._get_device(device)

        # Load YOLOv5-face detector
        self.detector = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(detector_path),
        )
        self.detector.to(self.device)
        self.detector.eval()

        # Load embedder
        self.embedder = PetFaceEmbedder()
        self.embedder.load_state_dict(
            torch.load(embedder_path, map_location=self.device)
        )
        self.embedder.to(self.device)
        self.embedder.eval()

        # Load gallery if provided
        self.gallery = {}
        if gallery_path and Path(gallery_path).exists():
            self.load_gallery(gallery_path)

        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _get_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def detect_faces(self, image: Image.Image) -> list[dict]:
        """Detect pet faces in image."""
        results = self.detector(image)
        detections = []

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "class": int(cls),
            })

        return detections

    def crop_face(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        margin: float = 0.1,
    ) -> Image.Image:
        """Crop face with margin."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        # Add margin
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(image.width, int(x2 + w * margin))
        y2 = min(image.height, int(y2 + h * margin))

        face = image.crop((x1, y1, x2, y2))
        return face.resize((self.image_size, self.image_size))

    def preprocess(self, face: Image.Image) -> torch.Tensor:
        """Preprocess face image for embedding."""
        face = np.array(face).astype(np.float32) / 255.0
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
        face = face.to(self.device)
        face = (face - self.mean) / self.std
        return face

    @torch.no_grad()
    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Get normalized embedding."""
        return self.embedder(face_tensor).squeeze(0)

    def match_gallery(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[str, float]:
        """Find best match in gallery."""
        if not self.gallery:
            return "unknown", 0.0

        best_match = "unknown"
        best_score = -1.0

        for name, stored_embeddings in self.gallery.items():
            for stored_emb in stored_embeddings:
                score = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    stored_emb.unsqueeze(0),
                ).item()

                if score > best_score:
                    best_score = score
                    best_match = name

        if best_score >= threshold:
            return best_match, best_score

        return "unknown", best_score

    def recognize(
        self,
        image: str | Path | Image.Image,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Full recognition pipeline."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        results = []
        detections = self.detect_faces(image)

        for det in detections:
            face = self.crop_face(image, det["bbox"])
            face_tensor = self.preprocess(face)
            embedding = self.get_embedding(face_tensor)
            identity, similarity = self.match_gallery(embedding, threshold)

            results.append({
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                "identity": identity,
                "similarity": similarity,
                "embedding": embedding.cpu(),
            })

        return results

    def register_pet(
        self,
        name: str,
        images: list[str | Path | Image.Image],
        detect_face: bool = True,
    ):
        """Register a pet with multiple images."""
        embeddings = []

        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")

            if detect_face:
                detections = self.detect_faces(img)
                if detections:
                    face = self.crop_face(img, detections[0]["bbox"])
                else:
                    print(f"No face detected, using full image")
                    face = img.resize((self.image_size, self.image_size))
            else:
                face = img.resize((self.image_size, self.image_size))

            face_tensor = self.preprocess(face)
            emb = self.get_embedding(face_tensor)
            embeddings.append(emb)

        if name in self.gallery:
            self.gallery[name].extend(embeddings)
        else:
            self.gallery[name] = embeddings

        print(f"Registered '{name}' with {len(embeddings)} embeddings")
        print(f"Gallery now contains {len(self.gallery)} pets")

    def save_gallery(self, path: str | Path):
        """Save gallery to file."""
        # Convert to CPU tensors for saving
        gallery_cpu = {
            name: [emb.cpu() for emb in embeddings]
            for name, embeddings in self.gallery.items()
        }
        torch.save(gallery_cpu, path)
        print(f"Gallery saved to {path}")

    def load_gallery(self, path: str | Path):
        """Load gallery from file."""
        gallery_cpu = torch.load(path, map_location="cpu")
        self.gallery = {
            name: [emb.to(self.device) for emb in embeddings]
            for name, embeddings in gallery_cpu.items()
        }
        print(f"Loaded gallery with {len(self.gallery)} pets")

    def compare_faces(
        self,
        image1: str | Path | Image.Image,
        image2: str | Path | Image.Image,
    ) -> tuple[bool, float]:
        """Compare two face images directly."""
        results1 = self.recognize(image1)
        results2 = self.recognize(image2)

        if not results1 or not results2:
            return False, 0.0

        emb1 = results1[0]["embedding"].to(self.device)
        emb2 = results2[0]["embedding"].to(self.device)

        similarity = F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0),
        ).item()

        return similarity > 0.5, similarity


class ONNXPetRecognition:
    """ONNX-based inference for deployment."""

    def __init__(
        self,
        onnx_path: str | Path,
        image_size: int = 160,
    ):
        import onnxruntime as ort

        self.image_size = image_size
        self.session = ort.InferenceSession(str(onnx_path))

        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def preprocess(self, face: Image.Image) -> np.ndarray:
        """Preprocess face for ONNX inference."""
        face = face.resize((self.image_size, self.image_size))
        face = np.array(face).astype(np.float32) / 255.0
        face = face.transpose(2, 0, 1)[np.newaxis, ...]
        face = (face - self.mean) / self.std
        return face.astype(np.float32)

    def get_embedding(self, face: Image.Image) -> np.ndarray:
        """Get embedding using ONNX runtime."""
        input_tensor = self.preprocess(face)
        outputs = self.session.run(None, {"image": input_tensor})
        return outputs[0][0]

    def compare(self, face1: Image.Image, face2: Image.Image) -> float:
        """Compare two faces."""
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(similarity)


# Example usage
if __name__ == "__main__":
    # Example: Initialize pipeline
    # pipeline = PetRecognitionPipeline(
    #     detector_path="yolov5_pet_face.pt",
    #     embedder_path="outputs/pet_embedder.pt",
    # )

    # Register pets
    # pipeline.register_pet("Max", ["max_1.jpg", "max_2.jpg", "max_3.jpg"])
    # pipeline.register_pet("Luna", ["luna_1.jpg", "luna_2.jpg"])
    # pipeline.save_gallery("my_pets.pt")

    # Recognize
    # results = pipeline.recognize("test_photo.jpg")
    # for r in results:
    #     print(f"Found {r['identity']} (similarity: {r['similarity']:.2f})")

    print("Pet Recognition Pipeline ready!")
    print("See docstrings for usage examples.")
