# File: ConditionalGAN.py
# Description: A robust Conditional GAN with a dynamic architecture, stabilized
#              training, self-attention, and automatic checkpointing.

import torch
import torch.nn as nn
import torchvision.transforms as T
import math
import os
import glob

from transformers import AutoTokenizer, AutoModel


def supported_hyperparameters():
    """Returns the hyperparameters supported by this model."""
    # Adding a dummy hyperparameter to easily reset the framework's trial history.
    return {'lr', 'momentum', 'version'}


class SelfAttention(nn.Module):
    """
    A Self-Attention layer to help the model learn spatial relationships.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key).softmax(dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class Net(nn.Module):
    class TextEncoder(nn.Module):
        def __init__(self, out_size=128):
            super().__init__()
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.text_linear = nn.Linear(768, out_size)
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            return self.text_linear(outputs.last_hidden_state.mean(dim=1))

    class Generator(nn.Module):
        def __init__(self, noise_dim=100, text_embedding_dim=128, image_channels=3, image_size=64):
            super().__init__()
            self.model = nn.Sequential(
                nn.ConvTranspose2d(noise_dim + text_embedding_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                SelfAttention(256),  # Added attention
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, noise, text_embedding):
            # The noise vector needs spatial dimensions to be compatible with the ConvTranspose2d layer.
            # We unsqueeze it twice to add a 1x1 height and width.
            noise = noise.unsqueeze(-1).unsqueeze(-1)
            combined_input = torch.cat([noise, text_embedding.unsqueeze(-1).unsqueeze(-1)], dim=1)
            return self.model(combined_input)

    class Discriminator(nn.Module):
        def __init__(self, text_embedding_dim=128, image_channels=3, image_size=64):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(image_channels + text_embedding_dim, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(256),  # Added attention
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            )

        def forward(self, image, text_embedding):
            text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image.size(2),
                                                                                        image.size(3))
            combined_input = torch.cat([image, text_embedding_expanded], dim=1)
            return self.main(combined_input)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.noise_dim = 100
        self.text_embedding_dim = 128
        self.epoch_counter = 0
        self.model_name = "ConditionalGAN"

        image_channels, image_size = in_shape[1], in_shape[2]
        self.text_encoder = self.TextEncoder(out_size=self.text_embedding_dim).to(device)
        self.generator = self.Generator(self.noise_dim, self.text_embedding_dim, image_channels, image_size).to(device)
        self.discriminator = self.Discriminator(self.text_embedding_dim, image_channels, image_size).to(device)

        self.checkpoint_dir = os.path.join("checkpoints", self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.load_checkpoint()

    def load_checkpoint(self):
        g_files = glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_generator_epoch_*.pth'))
        d_files = glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_discriminator_epoch_*.pth'))

        if g_files and d_files:
            latest_g = max(g_files, key=os.path.getctime)
            latest_d = max(d_files, key=os.path.getctime)
            print(f"Loading generator checkpoint: {latest_g}")
            print(f"Loading discriminator checkpoint: {latest_d}")
            self.generator.load_state_dict(torch.load(latest_g, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(latest_d, map_location=self.device))
            try:
                self.epoch_counter = int(os.path.basename(latest_g).split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                self.epoch_counter = 0
        else:
            print("No checkpoint found, starting from scratch.")

    def train_setup(self, prm):
        lr = 2e-4
        beta1 = 0.5
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, 'max', patience=50, factor=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d, 'max', patience=50, factor=0.5)

    def learn(self, train_data):
        self.train()
        for batch in train_data:
            real_images, text_prompts = batch
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            text_embeddings = self.text_encoder(text_prompts)

            # --- Train Discriminator ---
            self.optimizer_d.zero_grad()

            # Real batch
            real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device)
            output_real = self.discriminator(real_images, text_embeddings)
            loss_d_real = self.criterion(output_real, real_labels)

            # Fake batch
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_images = self.generator(noise, text_embeddings)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device)
            output_fake = self.discriminator(fake_images.detach(), text_embeddings)
            loss_d_fake = self.criterion(output_fake, fake_labels)

            # --- THIS IS THE FIX ---
            # Combine the real and fake loss before calling backward once.
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            # --- END OF FIX ---

            self.optimizer_d.step()

            # --- Train Generator ---
            self.optimizer_g.zero_grad()
            # We need to run the discriminator again on the fake images, but this time
            # we don't detach them, so gradients can flow back to the generator.
            output_g = self.discriminator(fake_images, text_embeddings)
            loss_g = self.criterion(output_g, real_labels)  # Generator wants the discriminator to think they are real

            loss_g.backward()
            self.optimizer_g.step()

    @torch.no_grad()
    def generate(self, text_prompts):
        self.eval()
        num_images = len(text_prompts)
        noise = torch.randn(num_images, self.noise_dim, device=self.device)
        text_embeddings = self.text_encoder(text_prompts)
        generated_images = self.generator(noise, text_embeddings)
        generated_images = (generated_images + 1) / 2
        return [T.ToPILImage()(img.cpu()) for img in generated_images]

    @torch.no_grad()
    def forward(self, images, **kwargs):
        batch_size = images.size(0)
        fixed_prompts_for_eval = [
            "a photo of a dog", "a painting of a car", "a smiling person"
        ]
        prompts_to_use = [fixed_prompts_for_eval[i % len(fixed_prompts_for_eval)] for i in range(batch_size)]

        output_dir = os.path.join("output_images", self.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        custom_prompts_to_generate = [
            "a smiling woman with blond hair",
            "a man wearing eyeglasses"
        ]
        if custom_prompts_to_generate:
            print(f"\n[Inference] Generating {len(custom_prompts_to_generate)} custom image(s)...")
            custom_images = self.generate(custom_prompts_to_generate)
            for i, img in enumerate(custom_images):
                save_path = os.path.join(output_dir,
                                         f"{self.model_name}_output_epoch_{self.epoch_counter}_image_{i + 1}.png")
                img.save(save_path)
                print(f"[Inference] Saved custom image to {save_path}")

        eval_images = self.generate(prompts_to_use)
        self.epoch_counter += 1

        if self.epoch_counter % 100 == 0:
            g_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_generator_epoch_{self.epoch_counter}.pth")
            d_path = os.path.join(self.checkpoint_dir,
                                  f"{self.model_name}_discriminator_epoch_{self.epoch_counter}.pth")
            torch.save(self.generator.state_dict(), g_path)
            torch.save(self.discriminator.state_dict(), d_path)
            print(f"\nSaved checkpoint to {g_path} and {d_path}")

        return eval_images, prompts_to_use
