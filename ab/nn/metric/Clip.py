# File: clip.py
# Location: ab/nn/metric/
# Description: This file defines the CLIPMetric class, conforming to the LEMUR framework's
# stateful metric pattern.

import torch
from PIL import Image

# Ensure these libraries are installed in your environment
# pip install transformers ftfy
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install the 'transformers' library for CLIP metric: pip install transformers")
try:
    import ftfy
except ImportError:
    raise ImportError("Please install the 'ftfy' library for CLIP metric: pip install ftfy")


# Use a specific, widely-used version of the CLIP model
MODEL_NAME = "openai/clip-vit-base-patch32"

# A global cache for the model and processor. This is a crucial optimization
# to prevent reloading the large CLIP model from disk on every evaluation call.
_clip_model_cache = {}

def _get_clip_model(device):
    """
    Helper function to load or retrieve the cached CLIP model and processor.
    """
    if 'model' not in _clip_model_cache:
        print(f"\n[CLIPMetric] Caching CLIP model '{MODEL_NAME}' for the first time...")
        try:
            model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
            processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            _clip_model_cache['model'] = model
            _clip_model_cache['processor'] = processor
            print("[CLIPMetric] Model cached successfully.")
        except Exception as e:
            raise IOError(f"Failed to download CLIP model from Hugging Face. Check network connection. Error: {e}")

    # Ensure the cached model is on the correct device for the current run
    _clip_model_cache['model'].to(device)
    return _clip_model_cache['model'], _clip_model_cache['processor']


class CLIPMetric:
    """
    A stateful metric class for calculating the CLIP score, which measures
    the semantic similarity between generated images and their text prompts.
    """
    def __init__(self, out_shape=None, device=None):
        """
        Initializes the metric.
        Args:
            out_shape: Unused for this metric, but kept for framework compatibility.
            device: The torch.device to run calculations on.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        """
        Resets the internal state, clearing all accumulated scores.
        This is called by the framework at the beginning of each evaluation phase.
        """
        self.similarity_scores = []
        self.num_samples = 0

    def __call__(self, preds, labels):
        """
        Processes a batch of predictions and labels to update the metric state.

        Args:
            preds (list[PIL.Image.Image]): A batch of generated images.
            labels (list[str]): The corresponding batch of text prompts.
        """
        if not preds or not labels:
            return

        # Get the cached CLIP model and processor
        model, processor = _get_clip_model(self.device)
        model.eval()

        with torch.no_grad():
            # Preprocess the images and text prompts
            inputs = processor(
                text=labels,
                images=preds,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move inputs to the correct device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # L2-normalize the embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Calculate the cosine similarity
            batch_scores = (image_embeds * text_embeds).sum(dim=-1)

            # Accumulate the scores and the sample count
            self.similarity_scores.append(batch_scores.sum().item())
            self.num_samples += len(preds)

    def result(self):
        """
        Calculates and returns the final, primary metric score.
        """
        if self.num_samples == 0:
            return 0.0

        avg_score = sum(self.similarity_scores) / self.num_samples
        return float(avg_score * 100)

    def get_all(self):
        """
        Returns a dictionary of all computed metrics.
        """
        final_score = self.result()
        return {'CLIP_Score': final_score}

def create_metric(out_shape=None, device=None):
    """
    Factory function used by the LEMUR framework to instantiate the metric class.
    """
    return CLIPMetric(out_shape=out_shape, device=device)
