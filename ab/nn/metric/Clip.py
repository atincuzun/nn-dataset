# File: Clip.py
# Location: ab/nn/metric/
# Note the capital 'C' in the filename to match the collection string.

import torch
from PIL import Image

# Ensure these libraries are installed in your environment
# pip install transformers ftfy
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install 'transformers' and 'ftfy': pip install transformers ftfy")

MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model_cache = {}

def _get_clip_model(device):
    """ Helper function to load or retrieve the cached CLIP model and processor. """
    if 'model' not in _clip_model_cache:
        print(f"\n[CLIPMetric] Caching CLIP model '{MODEL_NAME}' for the first time...")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _clip_model_cache['model'] = model
        _clip_model_cache['processor'] = processor
        print("[CLIPMetric] Model cached successfully.")
    _clip_model_cache['model'].to(device)
    return _clip_model_cache['model'], _clip_model_cache['processor']

class CLIPMetric:
    """ A stateful metric class for calculating the CLIP score. """
    def __init__(self, out_shape=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        """ Resets the internal state for a new evaluation. """
        self.similarity_scores = []
        self.num_samples = 0

    def __call__(self, preds, labels):
        """ Processes a batch of generated images and text prompts. """
        if not preds or not labels: return
        model, processor = _get_clip_model(self.device)
        model.eval()
        with torch.no_grad():
            inputs = processor(text=labels, images=preds, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            batch_scores = (image_embeds * text_embeds).sum(dim=-1)
            self.similarity_scores.append(batch_scores.sum().item())
            self.num_samples += len(preds)

    def result(self):
        """ Calculates and returns the final, primary metric score. """
        if self.num_samples == 0: return 0.0
        avg_score = sum(self.similarity_scores) / self.num_samples
        return float(avg_score * 100)

    def get_all(self):
        """ Returns a dictionary of all computed metrics. """
        final_score = self.result()
        return {'CLIP_Score': final_score}

def create_metric(out_shape=None, device=None):
    """ Factory function used by the LEMUR framework to instantiate the metric. """
    return CLIPMetric(out_shape=out_shape, device=device)