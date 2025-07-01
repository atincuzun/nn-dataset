# File: VQGAN_CLIP.py
# Location: ab/nn/nn/
# Description: A robust and simpler text-to-image model using a pre-trained VQGAN
# and a trainable Transformer, designed for stability within the LEMUR framework.

import os
import random
import itertools
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Dependencies required by the model. Ensure they are installed in your environment.
# pip install transformers accelerate
from transformers import VQGANModel, CLIPTextModel, CLIPTokenizer, GPT2LMHeadModel, GPT2Config

# --- Import metric classes for type checking in the evaluate method ---
try:
    from ab.nn.metric.clip import CLIPMetric
    # FIDMetric is no longer used in this version
    # from ab.nn.metric.fid import FIDMetric
except ImportError:
    print("[VQGAN_CLIP WARN] Could not import metric classes. Evaluation might fail if run in the framework.")
    CLIPMetric = None
    # FIDMetric = None # No longer needed


class Net(nn.Module):
    """
    A text-to-image model using a frozen, pre-trained VQGAN and a trainable
    Transformer (GPT-2 style) to generate images from text prompts.
    """

    def __init__(self, in_shape, out_shape, prm, device):
        """
        Initializes the complete model.
        """
        super().__init__()

        # --- Self-Configuration Logic ---
        # Detects if the framework is in random search mode and applies a valid default config.
        is_random_search = not isinstance(prm.get('image_size'), int)
        if is_random_search:
            print("\n[VQGAN_CLIP WARNING] Framework in default search mode. Applying internal configuration.")
            default_prm = {
                'lr': prm.get('lr', 5e-5),
                'batch': prm.get('batch', 4),
                'momentum': prm.get('momentum', 0.9),
                'image_size': 256,  # VQGAN works best with 256x256 images
                'transformer_layers': 8,
                'transformer_heads': 8,
                'transformer_embed_dim': 768,
            }
            prm = default_prm

        self.prm = prm
        self.device = device
        self.in_shape = in_shape

        print("[VQGAN_CLIP INFO] Initializing model with parameters:", json.dumps(self.prm, indent=2, default=str))

        # --- 1. Load Frozen, Pre-trained Components ---
        print("[VQGAN_CLIP INFO] Loading pre-trained VQGAN model...")
        self.vqgan = VQGANModel.from_pretrained("CompVis/vqgan-f16-16384").to(device)
        self.vqgan.eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False

        print("[VQGAN_CLIP INFO] Loading pre-trained CLIP text model...")
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_model.eval()
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

        # --- 2. Initialize the Trainable Transformer ---
        # This is the only part of the model that will be trained.
        # It's a GPT-2 style model that will learn to generate VQGAN codes.
        print("[VQGAN_CLIP INFO] Initializing trainable Transformer...")
        # The vocabulary size is the number of codes in the VQGAN's codebook.
        config = GPT2Config(
            vocab_size=self.vqgan.config.num_embeddings,
            n_positions=self.vqgan.config.num_patches + 1,
            n_layer=self.prm['transformer_layers'],
            n_head=self.prm['transformer_heads'],
            n_embd=self.prm['transformer_embed_dim'],
            # We add the CLIP embedding dimension for cross-attention
            add_cross_attention=True,
        )
        self.transformer = GPT2LMHeadModel(config).to(device)

        # A projection layer to match CLIP's output dimension to the Transformer's
        self.clip_projection = nn.Linear(self.clip_text_model.config.hidden_size, config.n_embd).to(device)

    def train_setup(self, prm):
        """Initializes the optimizer and loss criterion for the trainable parts."""
        # We only train the Transformer and the projection layer.
        trainable_params = itertools.chain(self.transformer.parameters(), self.clip_projection.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.prm['lr'])
        # The task is to predict the next code, so we use CrossEntropyLoss.
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, train_data):
        """Implements the training pipeline for one epoch."""
        self.transformer.train()
        self.clip_projection.train()
        total_loss = 0.0

        for real_images, text_prompts in train_data:
            real_images = real_images.to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                # 1. Encode real images into discrete VQGAN codes (our target sequence)
                # We expect the dataloader to normalize images to [-1, 1]
                latents = self.vqgan.encode(real_images).latents
                image_codes = self.vqgan.quantize(latents).indices.flatten(1)

                # 2. Encode text prompts using CLIP
                text_inputs = self.clip_tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True).to(
                    self.device)
                text_features = self.clip_text_model(**text_inputs).last_hidden_state

                # 3. Project CLIP features to match transformer dimension
                encoder_hidden_states = self.clip_projection(text_features)

            # Prepare inputs for the Transformer
            # The target is the sequence of image codes
            labels = image_codes
            # The input to the transformer is the same sequence, shifted by one
            input_ids = torch.cat((torch.full((labels.size(0), 1), self.transformer.config.vocab_size - 1,
                                              dtype=torch.long, device=self.device), labels[:, :-1]), dim=1)

            # 4. Forward pass through the Transformer
            outputs = self.transformer(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                labels=labels
            )

            # 5. Calculate loss
            loss = outputs.loss
            assert loss is not None, "Transformer returned None for loss."

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_data) if len(train_data) > 0 else 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=256):
        """Helper method to generate images from text prompts."""
        self.eval()

        # 1. Get text embeddings from CLIP
        text_inputs = self.clip_tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True).to(
            self.device)
        text_features = self.clip_text_model(**text_inputs).last_hidden_state
        encoder_hidden_states = self.clip_projection(text_features)

        # 2. Generate image codes using the Transformer autoregressively
        # Start with a beginning-of-sequence token
        input_ids = torch.full((len(text_prompts), 1), self.transformer.config.vocab_size - 1, dtype=torch.long,
                               device=self.device)

        generated_codes = self.transformer.generate(
            input_ids,
            max_length=self.vqgan.config.num_patches + 1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            encoder_hidden_states=encoder_hidden_states
        )

        # Remove the starting token
        generated_codes = generated_codes[:, 1:]

        # 3. Decode the generated codes into an image with the VQGAN
        latents = self.vqgan.dequantize(generated_codes)
        images = self.vqgan.decode(latents).sample

        # Post-process images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray((image * 255).round().astype("uint8")) for image in images]
        return pil_images

    def evaluate(self, test_data, metric):
        """The core evaluation interface for the LEMUR framework, simplified for CLIP score."""
        print(f"\n[Evaluation] Starting evaluation phase with metric: {type(metric).__name__}...")
        self.eval()
        metric.reset()
        eval_samples_limit = 64  # Keep this lower as generation can be slow
        processed_samples = 0

        with torch.no_grad():
            for i, (real_images, text_prompts) in enumerate(test_data):
                if processed_samples >= eval_samples_limit: break

                # Generate fake images from the text prompts
                fake_images = self.generate(text_prompts)

                # This method now only expects to be called with a CLIPMetric object.
                # The framework ensures this when you run with `..._clip` in the collection name.
                if CLIPMetric is not None and isinstance(metric, CLIPMetric):
                    metric(fake_images, text_prompts)
                else:
                    print(f"[Evaluation WARN] Metric type '{type(metric).__name__}' is not CLIPMetric. Skipping.")

                processed_samples += len(fake_images)

        print(f"[Evaluation] Processed {processed_samples} images for scoring.")
        results = metric.get_all()
        print(f"[Evaluation] Finished. Results: {results}")

        # The framework expects a single float value for optimization.
        return metric.result()


def supported_hyperparameters():
    """Declares the hyperparameters that can be tuned by the LEMUR framework."""
    return {'lr', 'momentum'}
