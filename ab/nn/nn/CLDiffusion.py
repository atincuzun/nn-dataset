print("\n\n--- EXECUTING THE LATEST CLDIFFUSION.PY ---\n\n")

# File: CLDiffusion.py
# Location: ab/nn/nn/
# Description: Simplified version of CLDiffusion using standard components from Hugging Face.

import itertools
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Import standard, pre-built components from Hugging Face libraries
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, AutoModel


class Net(nn.Module):
    """
    A simplified text-to-image diffusion model.
    This version uses standard, pre-built components from the diffusers library
    for clarity and simplicity.
    """

    class TextEncoder(nn.Module):
        """Encodes text prompts into embeddings."""

        def __init__(self, out_size=768):
            super().__init__()
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.text_linear = nn.Linear(768, out_size)

            # Freeze the pre-trained text model for simpler fine-tuning
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            return self.text_linear(outputs.last_hidden_state)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # --- BATCH SIZE SAFEGUARD ---
        # This code block prevents Out-of-Memory errors by capping the batch size.
        MAX_SAFE_BATCH_SIZE = 2
        requested_batch_size = self.prm.get('batch', MAX_SAFE_BATCH_SIZE)
        if requested_batch_size > MAX_SAFE_BATCH_SIZE:
            print(
                f"\n[CLDiffusion WARNING] Requested batch size {requested_batch_size} is too high and may cause OOM errors.")
            print(f"[CLDiffusion WARNING] Overriding batch size to {MAX_SAFE_BATCH_SIZE}.\n")
            self.prm['batch'] = MAX_SAFE_BATCH_SIZE
        # --- END OF SAFEGUARD ---

        # 1. VAE (Autoencoder): Using a standard pre-trained VAE.
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

        # 2. Text Encoder
        self.text_encoder = self.TextEncoder(out_size=prm.get('cross_attention_dim', 768)).to(device)

        # 3. UNet: The core noise-prediction model.
        self.unet = UNet2DConditionModel(
            sample_size=64,  # The VAE produces 64x64 latents for 512x512 images
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(256, 512, 1024),
            cross_attention_dim=prm.get('cross_attention_dim', 768)
        ).to(device)

        # 4. Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )

        self.vae.requires_grad_(False)

    def train_setup(self, prm):
        trainable_params = itertools.chain(self.unet.parameters(), self.text_encoder.text_linear.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.prm.get('lr', 1e-5)
        )
        self.criterion = nn.MSELoss()

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        for images, text_prompts in train_data:
            self.optimizer.zero_grad()

            latents = self.vae.encode(images.to(self.device)).latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=self.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            text_embeddings = self.text_encoder(text_prompts)
            noise_pred = self.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

            loss = self.criterion(noise_pred, noise)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_data) if train_data else 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=50):
        self.eval()
        text_embeddings = self.text_encoder(text_prompts)
        latents = torch.randn(
            (len(text_prompts), self.unet.config.in_channels, self.unet.config.sample_size,
             self.unet.config.sample_size),
            device=self.device
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(
                sample=latents,
                timestep=t,
                encoder_hidden_states=text_embeddings
            ).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((img * 255).astype(np.uint8)) for img in images]

    def evaluate(self, test_data, metric):
        self.eval()
        metric.reset()
        for images, prompts in test_data:
            generated_images = self.generate(prompts)
            metric(generated_images, prompts)
            break
        return metric.result()


def supported_hyperparameters():
    return {'lr', 'momentum'}