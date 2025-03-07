from pathlib import Path
import time

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.post_processing.stepwise_handler import StepwiseHandler
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.model_saver import ModelSaver
from mflux.weights.weight_handler import WeightHandler
from mflux.weights.weight_handler_lora import WeightHandlerLoRA
from mflux.weights.weight_util import WeightUtil


class Flux1(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model_config = model_config

        # Load and initialize the tokenizers from disk, huggingface cache, or download from huggingface
        tokenizers = TokenizerHandler(model_config.model_name, self.model_config.max_sequence_length, local_path)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        self.vae = VAE()
        self.transformer = Transformer(model_config)
        self.t5_text_encoder = T5Encoder()
        self.clip_text_encoder = CLIPEncoder()

        # Set the weights and quantize the model
        weights = WeightHandler.load_regular_weights(repo_id=model_config.model_name, local_path=local_path)
        self.bits = WeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=self.vae,
            transformer=self.transformer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # Set LoRA weights
        lora_weights = WeightHandlerLoRA.load_lora_weights(transformer=self.transformer, lora_files=lora_paths, lora_scales=lora_scales)  # fmt:off
        WeightHandlerLoRA.set_lora_weights(transformer=self.transformer, loras=lora_weights)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config = Config(),
        stepwise_output_dir: Path = None,
    ) -> GeneratedImage:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))
        stepwise_handler = StepwiseHandler(
            flux=self,
            config=config,
            seed=seed,
            prompt=prompt,
            time_steps=time_steps,
            output_dir=stepwise_output_dir,
        )

        # Start timing the generation process
        start_time = time.time()

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(seed, config, self.vae)

        # 2. Embed the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder(clip_tokens)

        for gen_step, t in enumerate(time_steps, 1):
            try:
                # 3.t Predict the noise
                noise = self.transformer.predict(
                    t=t,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    hidden_states=latents,
                    config=config,
                )

                # 4.t Take one denoise step
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents += noise * dt

                # Handle stepwise output if enabled
                stepwise_handler.process_step(gen_step, latents)

                # Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                stepwise_handler.handle_interruption()
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        # Calculate generation time
        generation_time = time.time() - start_time
        print(f"\nGeneration time (excluding model loading): {generation_time:.2f} seconds")

        # 5. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            init_image_path=config.init_image_path,
            init_image_strength=config.init_image_strength,
            config=config,
        )

    @staticmethod
    def from_alias(alias: str, quantize: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_alias(alias),
            quantize=quantize,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)

    def freeze(self, **kwargs):
        self.vae.freeze()
        self.transformer.freeze()
        self.t5_text_encoder.freeze()
        self.clip_text_encoder.freeze()
