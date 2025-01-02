from mflux import Flux1, Config, ModelConfig
import mlx.core as mx
from pathlib import Path
import zipfile
from mflux.dreambooth.state.zip_util import ZipUtil
import shutil

# Setup paths
output_dir = Path("./mflux-training/training_outputs/output_20250102_144552")
checkpoint_dir = output_dir / "_checkpoints"
checkpoint_zip = checkpoint_dir / "0000510_checkpoint.zip"
temp_dir = checkpoint_dir / "0000510_checkpoint"
lora_path = temp_dir / "0000510_adapter.safetensors"

# Extract checkpoint and convert to LoRA adapter
print(f"Extracting checkpoint from: {checkpoint_zip}")
if checkpoint_zip.exists():
    # Create temp directory if needed
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract using ZipUtil (same as in tests)
    ZipUtil.extract_all(
        zip_path=str(checkpoint_zip),
        output_dir=str(temp_dir)
    )
    
    if not lora_path.exists():
        print("\nAvailable files in extracted checkpoint:")
        for file in temp_dir.glob("*"):
            print(f"- {file.name}")
        raise FileNotFoundError(f"LoRA adapter not found at: {lora_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_zip}")

# Load base model with LoRA
flux = Flux1(
    model_config=ModelConfig.FLUX1_DEV,
    quantize=4,
    lora_paths=[str(lora_path)],
    lora_scales=[1.0]
)

# Define different prompts with enhanced quality descriptors
prompts = [
    "Ultra-detailed luxury timepiece, masterful product photography, sharp focus, 8k resolution, professional studio lighting, crystal clear details, award-winning product shot",
    "Hyper-realistic watch photography, extreme detail and clarity, perfect focus, premium product shot, ultra-sharp macro details, masterful lighting setup, 8k quality",
]

# Generate multiple images with different prompts and higher quality settings
base_config = Config(
    num_inference_steps=40,  # Increased from 20 for better quality
    height=832,
    width=832,
    guidance=7.0  # Increased from 3.0 for stronger adherence to prompt
)

# Try different seeds for each prompt
seeds = [42, 123, 456, 789]  # Multiple seeds per prompt
for prompt in prompts:
    for seed in seeds:
        print(f"\nGenerating image with seed {seed} and prompt: {prompt}")
        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            config=base_config
        )
        output_path = f"test_output_seed{seed}.png"
        image.save(output_path)
        print(f"Saved as {output_path}")

print("\nAll images generated successfully!")

# Optional: Cleanup
if temp_dir.exists():
    shutil.rmtree(temp_dir)