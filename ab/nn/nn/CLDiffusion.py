# File: CLDiffusion.py
# Location: ab/nn/nn/
# Description: The final, self-contained, end-to-end text-to-image model module
# for the LEMUR framework, including a correctly implemented dual-metric
# evaluation method for both FID and CLIP scores.

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
# pip install diffusers transformers
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, AutoModel


class Net(nn.Module):
    """
    A single, self-contained module for a text-to-image diffusion model.
    This class encapsulates all necessary components (VAE, TextEncoder, U-Net),
    includes self-configuration logic, and a correctly implemented evaluation
    method for both FID and CLIP scores.
    """

    # ===================================================================
    # │ NESTED MODEL DEFINITIONS (VAE and TextEncoder)                  │
    # ===================================================================
    class VAE(nn.Module):
        """ Full VAE architecture. """

        def __init__(self, output_channels=3, img_size=128, num_channels=32, num_layers=3, latent_variable_channels=4,
                     num_layers_per_block=2, final_act='tanh'):
            super().__init__()
            self.output_channels, self.img_size, self.num_channels, self.num_layers, self.latent_variable_channels, self.num_layers_per_block = output_channels, img_size, num_channels, num_layers, latent_variable_channels, num_layers_per_block
            dim = int(self.img_size * (0.5 ** self.num_layers));
            self.latent_shape = (self.latent_variable_channels, dim, dim)
            encoders_list, batch_norms_enc_list, encoder_res_list, channel_sizes = [], [], [], [output_channels]
            for i in range(self.num_layers):
                channel_sizes.append(self.num_channels if i == 0 else channel_sizes[-1] * 2)
                enc_list_in_block = [
                    nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], 4, 2, 1) if j == 0 else nn.Conv2d(
                        channel_sizes[i + 1], channel_sizes[i + 1], 4, 1, 'same') for j in range(num_layers_per_block)]
                encoders_list.append(nn.ModuleList(enc_list_in_block));
                encoder_res_list.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], 4, 2, 1));
                batch_norms_enc_list.append(nn.BatchNorm2d(channel_sizes[i + 1]))
            self.conv_encoders, self.encoder_res, self.batch_norms_enc = nn.ModuleList(encoders_list), nn.ModuleList(
                encoder_res_list), nn.ModuleList(batch_norms_enc_list)
            self.conv_mean, self.conv_std = nn.Conv2d(channel_sizes[-1], latent_variable_channels, 4, 1,
                                                      "same"), nn.Conv2d(channel_sizes[-1], latent_variable_channels, 4,
                                                                         1, "same")
            self.first_upsample = nn.ModuleList([nn.Conv2d(latent_variable_channels, channel_sizes[-1], 4, 1, 'same'),
                                                 nn.Conv2d(channel_sizes[-1], channel_sizes[-1], 4, 1, 'same')])
            upsamplers_list, paddings_list, conv_decoders_list, batch_norms_dec_list, decoder_res_list = [], [], [], [], []
            for i in reversed(range(2, len(channel_sizes))):
                upsamplers_list.append(nn.UpsamplingNearest2d(scale_factor=2));
                paddings_list.append(nn.ReplicationPad2d(1))
                conv_layers = [nn.Conv2d(channel_sizes[i], channel_sizes[i - 1], 3, 1) if j == 0 else nn.Conv2d(
                    channel_sizes[i - 1], channel_sizes[i - 1], 3, 1, "same") for j in range(num_layers_per_block)]
                decoder_res_list.append(nn.Conv2d(channel_sizes[i], channel_sizes[i - 1], 3, 1));
                conv_decoders_list.append(nn.ModuleList(conv_layers));
                batch_norms_dec_list.append(nn.BatchNorm2d(channel_sizes[i - 1], 1e-3))
            upsamplers_list.append(nn.UpsamplingNearest2d(scale_factor=2));
            paddings_list.append(nn.ReplicationPad2d(1));
            conv_decoders_list.append(nn.Conv2d(channel_sizes[1], output_channels, 3, 1))
            self.upsamplers, self.paddings, self.conv_decoders, self.batch_norms_dec, self.res_decoders = nn.ModuleList(
                upsamplers_list), nn.ModuleList(paddings_list), nn.ModuleList(conv_decoders_list), nn.ModuleList(
                batch_norms_dec_list), nn.ModuleList(decoder_res_list)
            self.leakyrelu, self.relu, self.final_act_fn = nn.LeakyReLU(
                0.2), nn.ReLU(), nn.Sigmoid() if final_act == "sigmoid" else nn.Tanh() if final_act == "tanh" else lambda \
                x: x

        def encode(self, x):
            for i in range(len(self.conv_encoders)):
                res = self.encoder_res[i](x)
                for j, layer in enumerate(self.conv_encoders[i]): x = layer(x); x = self.leakyrelu(x) if j != len(
                    self.conv_encoders[i]) - 1 else x
                x = self.relu(self.batch_norms_enc[i](x) + res)
            return self.conv_mean(x), self.conv_std(x)

        def decode(self, z):
            for layer in self.first_upsample: z = self.leakyrelu(layer(z))
            for i in range(len(self.upsamplers) - 1):
                z = self.paddings[i](self.upsamplers[i](z));
                res = self.res_decoders[i](z)
                for j, layer in enumerate(self.conv_decoders[i]): z = layer(z); z = self.leakyrelu(z) if j != len(
                    self.conv_decoders[i]) - 1 else z
                z += res
            return self.final_act_fn(self.conv_decoders[-1](self.paddings[-1](self.upsamplers[-1](z))))

    class TextEncoder(nn.Module):
        """ Full TextEncoder architecture. """

        def __init__(self, out_size=1280):
            super().__init__()
            self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            for name, params in self.text_model.named_parameters():
                params.requires_grad = "transformer.layer.5" in name
            self.text_linear_layer = nn.Linear(768, out_size)

        def forward(self, text):
            device = self.text_linear_layer.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            text_embeddings = self.text_model(inputs.input_ids.to(device),
                                              attention_mask=inputs.attention_mask.to(device)).last_hidden_state
            return self.text_linear_layer(text_embeddings), inputs.attention_mask.to(device)

    def __init__(self, in_shape, out_shape, prm, device):
        """
        Initializes the complete model. Includes logic to apply a default
        configuration if the framework provides random search parameters.
        """
        super().__init__()

        is_random_search = not isinstance(prm.get('image_size'), int)

        if is_random_search:
            print("\n[CLDiffusion WARNING] Framework in default search mode. Applying internal configuration.")

            default_prm = {
                'lr': prm.get('lr', 1e-5),
                'batch': prm.get('batch', 4),
                'momentum': prm.get('momentum', 0.9),
                'image_size': 128,
                'num_train_timesteps': 1000,
                'scale_factor': 0.5,
                'vae_latent_channels': 4,
                'vae_num_layers': 3,
                'layers_per_block': 2,
                'block_out_channels': [64, 128, 256],
                'down_block_types': ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                'up_block_types': ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
                'cross_attention_dim': 1280,
            }
            prm = default_prm

        self.prm = prm
        self.device = device
        self.in_shape = in_shape

        print("[CLDiffusion INFO] Initializing model with parameters:", json.dumps(self.prm, indent=2, default=str))

        self.vae = self.VAE(img_size=self.prm['image_size'], num_layers=self.prm['vae_num_layers'],
                            latent_variable_channels=self.prm['vae_latent_channels']).to(device)
        self.text_model = self.TextEncoder(out_size=self.prm['cross_attention_dim']).to(device)

        self.unet = UNet2DConditionModel(
            sample_size=self.vae.latent_shape[1],
            in_channels=self.vae.latent_shape[0],
            out_channels=self.vae.latent_shape[0],
            down_block_types=tuple(self.prm['down_block_types']),
            up_block_types=tuple(self.prm['up_block_types']),
            block_out_channels=tuple(self.prm['block_out_channels']),
            layers_per_block=self.prm['layers_per_block'],
            cross_attention_dim=self.prm['cross_attention_dim']
        ).to(device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.prm['num_train_timesteps'],
                                             beta_schedule="squaredcos_cap_v2")
        self.scale_factor = self.prm['scale_factor']

    def train_setup(self, prm):
        """Initializes the optimizer and loss criterion."""
        all_parameters = itertools.chain(self.vae.parameters(), self.text_model.parameters(), self.unet.parameters())
        self.optimizer = torch.optim.SGD(all_parameters, lr=self.prm['lr'], momentum=self.prm.get('momentum', 0.9))
        self.criterion = nn.MSELoss()

    def learn(self, train_data):
        """Implements the training pipeline for one epoch."""
        self.train()
        total_loss = 0.0
        for imgs, text_prompts in train_data:
            imgs = imgs.to(self.device)
            self.optimizer.zero_grad()
            latents, _ = self.vae.encode(imgs)
            latents = latents * self.scale_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=self.device, dtype=torch.int64)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            text_embeddings, _ = self.text_model(text_prompts)

            unet_output = self.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=True
            )
            noise_pred = unet_output.sample

            loss = self.criterion(noise_pred, noise)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data) if len(train_data) > 0 else 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=50):
        """
        Helper method to generate images from a list of text prompts.
        Returns a list of PIL Images.
        """
        self.eval()
        latents = torch.randn(
            (len(text_prompts), self.unet.config.in_channels, self.vae.latent_shape[1], self.vae.latent_shape[2]),
            device=self.device,
        )
        text_embeddings, _ = self.text_model(text_prompts)
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(latents, t, text_embeddings, return_dict=False)[0]
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.scale_factor
        images = self.vae.decode(latents)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray((image * 255).round().astype("uint8")) for image in images]
        return pil_images

    def evaluate(self, test_data, metric):
        """
        This is the core evaluation interface for the LEMUR framework, updated to
        calculate both FID and CLIP scores.
        """
        try:
            from ab.nn.metric.fid import create_metric as create_fid_metric
            from ab.nn.metric.clip import create_metric as create_clip_metric
        except ImportError as e:
            raise ImportError(
                f"Could not import metric factories. Ensure fid.py and clip.py are in ab/nn/metric/. Error: {e}")

        print("\n[Evaluation] Starting evaluation phase for FID and CLIP scores...")
        self.eval()

        # Instantiate both metric trackers
        fid_metric = create_fid_metric(device=self.device)
        clip_metric = create_clip_metric(device=self.device)

        eval_samples_limit = 256
        print(f"[Evaluation] Will process up to {eval_samples_limit} samples.")

        with torch.no_grad():
            for i, (real_images, text_prompts) in enumerate(test_data):
                if fid_metric.num_samples >= eval_samples_limit:
                    break

                # 1. Generate fake images from the text prompts
                fake_images = self.generate(text_prompts)

                # 2. Feed the FID metric (fake vs. real images)
                fid_metric(fake_images, real_images)

                # 3. Feed the CLIP metric (fake images vs. text prompts)
                clip_metric(fake_images, text_prompts)

        print(f"[Evaluation] Processed {fid_metric.num_samples} images for scoring.")

        # Get the final dictionary of scores from both metrics
        fid_results = fid_metric.get_all()
        clip_results = clip_metric.get_all()

        # Combine the results into a single dictionary
        final_results = {**fid_results, **clip_results}

        print(f"[Evaluation] Finished. Results: {final_results}")

        return final_results


def supported_hyperparameters():
    """
    Declares the hyperparameters that can be tuned by the LEMUR framework.
    """
    return {'lr', 'momentum'}
