"""Export pet embedder model for mobile deployment."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import torch.nn as nn

from src.model import PetFaceEmbedder


def export_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int = 160,
    opset_version: int = 13,
):
    """Export to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    print(f"Exported ONNX model to: {output_path}")


def export_coreml(
    model: nn.Module,
    output_path: Path,
    image_size: int = 160,
):
    """Export to CoreML format for iOS."""
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed. Run: pip install coremltools")
        return

    model.eval()

    # Trace the model
    example_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 3, image_size, image_size),
                dtype=float,
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS15,
    )

    # Add metadata
    mlmodel.author = "Pet Recognition"
    mlmodel.short_description = "Pet face embedding model"
    mlmodel.input_description["image"] = "Pet face image (RGB, normalized)"
    mlmodel.output_description["embedding"] = "128-dim face embedding"

    mlmodel.save(str(output_path))
    print(f"Exported CoreML model to: {output_path}")


def export_torchscript(
    model: nn.Module,
    output_path: Path,
    image_size: int = 160,
):
    """Export to TorchScript for mobile."""
    model.eval()
    example_input = torch.randn(1, 3, image_size, image_size)

    # Script the model
    scripted_model = torch.jit.trace(model, example_input)

    # Optimize for mobile
    optimized_model = torch.jit.optimize_for_mobile(scripted_model)

    optimized_model._save_for_lite_interpreter(str(output_path))
    print(f"Exported TorchScript Lite model to: {output_path}")


def export_tflite(
    onnx_path: Path,
    output_path: Path,
):
    """Convert ONNX to TFLite (requires onnx2tf)."""
    try:
        import onnx2tf
    except ImportError:
        print("onnx2tf not installed. Run: pip install onnx2tf")
        print("Alternative: Use ai-edge-torch or ONNX Runtime Mobile")
        return

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_path.parent),
        output_file_name=output_path.stem,
        copy_onnx_input_output_names_to_tflite=True,
    )
    print(f"Exported TFLite model to: {output_path}")


def quantize_model(
    model: nn.Module,
    output_path: Path,
    image_size: int = 160,
):
    """Quantize model to INT8."""
    model.eval()

    # Dynamic quantization (simplest)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Save
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Exported quantized model to: {output_path}")

    # Print size comparison
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quant_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    print(f"Original size: {orig_size / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {quant_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export pet embedder for mobile")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (pet_embedder.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Input image size",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "coreml", "torchscript"],
        choices=["onnx", "coreml", "torchscript", "tflite", "quantized"],
        help="Export formats",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = PetFaceEmbedder()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Export to requested formats
    if "onnx" in args.formats:
        export_onnx(
            model,
            args.output_dir / "pet_embedder.onnx",
            args.image_size,
        )

    if "coreml" in args.formats:
        export_coreml(
            model,
            args.output_dir / "PetEmbedder.mlpackage",
            args.image_size,
        )

    if "torchscript" in args.formats:
        export_torchscript(
            model,
            args.output_dir / "pet_embedder.ptl",
            args.image_size,
        )

    if "tflite" in args.formats:
        onnx_path = args.output_dir / "pet_embedder.onnx"
        if not onnx_path.exists():
            export_onnx(model, onnx_path, args.image_size)
        export_tflite(
            onnx_path,
            args.output_dir / "pet_embedder.tflite",
        )

    if "quantized" in args.formats:
        quantize_model(
            model,
            args.output_dir / "pet_embedder_int8.pt",
            args.image_size,
        )

    print("\nExport complete!")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
