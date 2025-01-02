import os
import time
from mflux import Flux1, Config

# Base configuration
base_config = Config(
    num_inference_steps=4, 
    height=1024,
    width=1024,
)

# Initialize model
flux = Flux1.from_alias(
    alias="schnell",
    quantize=8,  # Using 8-bit quantization for speed
)

# Product photography prompts
prompts = [
    "Luxury perfume bottle on black marble, dramatic lighting, professional product photography, soft shadows, crystal clear glass, golden liquid inside, 8k, photorealistic",
    "Premium watch on leather surface, macro photography, dramatic side lighting, professional product photography, showing intricate details, luxury timepiece, 8k, photorealistic",
    "High-end leather bag on white surface, soft diffused lighting, professional studio setup, showing texture details, luxury fashion, 8k, photorealistic",
    "Designer sunglasses on reflective surface, dramatic lighting, professional product photography, showing material quality, luxury eyewear, 8k, photorealistic",
    "Premium fountain pen on dark wood, dramatic spot lighting, professional product photography, showing metallic details, luxury writing instrument, 8k, photorealistic",
    "Luxury jewelry box with gold accents, soft directional lighting, professional product photography, rich textures, premium packaging, 8k, photorealistic",
    "High-end wireless headphones on glass surface, dramatic rim lighting, professional product photography, showing premium materials, 8k, photorealistic",
    "Premium smartphone on metallic surface, dramatic lighting, professional product photography, showing premium build quality, 8k, photorealistic",
    "Luxury wine bottle and glass, moody lighting, professional product photography, showing label details and wine color, 8k, photorealistic",
    "Premium skincare product on marble, soft diffused lighting, professional product photography, showing packaging luxury, 8k, photorealistic",
    "Designer wallet on leather surface, dramatic side lighting, professional product photography, showing leather grain, 8k, photorealistic",
    "Luxury candle with custom packaging, atmospheric lighting, professional product photography, showing brand details, 8k, photorealistic",
    "Premium coffee beans scattered on dark marble, overhead shot, professional product photography, dramatic rim lighting, showing texture and freshness, 8k, photorealistic",
    "Luxury watch movement macro shot, extreme detail, professional product photography, dramatic side lighting, showing intricate mechanical details, 8k, photorealistic",
    "High-end makeup palette with metallic finishes, soft gradient lighting, professional beauty photography, showing color and texture detail, 8k, photorealistic",
    "Premium chocolate truffles on black slate, selective focus, professional food photography, moody lighting with highlights on texture, 8k, photorealistic",
    "Designer jewelry on mirror surface, split lighting setup, professional product photography, capturing gemstone brilliance, 8k, photorealistic"
]

# Generate images
for idx, prompt in enumerate(prompts):
    print(f"\nGenerating image {idx + 1}/{len(prompts)}")
    print(f"Prompt: {prompt}\n")
    
    image = flux.generate_image(
        seed=42 + idx,  # Different seed for each image
        prompt=prompt,
        config=base_config
    )
    
    # Save image
    output_path = f"images/luxury_product_{idx+1:02d}.png"
    image.save(path=output_path)
    print(f"Saved: {output_path}")
    
    # Small delay between generations
    time.sleep(1)

print("\nDataset generation complete!")
