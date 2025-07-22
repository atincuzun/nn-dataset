# File: Clip.py
# Location: ab/nn/metric/
# Description: This file defines the CLIPMetric class, with a safeguard
#              to prevent crashes from the framework's generic eval loop.

import torch
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install the 'transformers' library for CLIP metric: pip install transformers")
try:
    import ftfy
except ImportError:
    raise ImportError("Please install the 'ftfy' library for CLIP metric: pip install ftfy")

MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model_cache = {}


def _get_clip_model(device):
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

    _clip_model_cache['model'].to(device)
    return _clip_model_cache['model'], _clip_model_cache['processor']


class CLIPMetric:
    def __init__(self, out_shape=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.similarity_scores = []
        self.num_samples = 0

    def __call__(self, preds, labels):
        # --- CHANGE START: Add safeguard for incorrect data types ---
        # The framework's generic eval loop passes tensors instead of lists.
        # This check detects that and exits gracefully to prevent a crash.
        if isinstance(labels, torch.Tensor):
            print(
                "[CLIPMetric WARN] Invalid data types detected during evaluation. Skipping metric calculation to prevent a crash.")
            return
        # --- CHANGE END ---

        if not preds or not labels:
            return

        model, processor = _get_clip_model(self.device)
        model.eval()

        with torch.no_grad():
            inputs = processor(
                text=labels,
                images=preds,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = model(**inputs)

            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

            batch_scores = (image_embeds * text_embeds).sum(dim=-1)

            self.similarity_scores.append(batch_scores.sum().item())
            self.num_samples += len(preds)

    def result(self):
        if self.num_samples == 0:
            return 0.0
        avg_score = sum(self.similarity_scores) / self.num_samples
        return float(avg_score * 100)

    def get_all(self):
        final_score = self.result()
        return {'CLIP_Score': final_score}


def create_metric(out_shape=None, device=None):
    return CLIPMetric(out_shape=out_shape, device=device)