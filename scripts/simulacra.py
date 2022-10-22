import os
import torch
from torch.nn import functional as F, Module, Linear
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from clip import clip


class AestheticMeanPredictionLinearModel(Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = Linear(feats_in, 1)

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)


clip_model = clip.load("ViT-B/16",
                       jit=False,
                       device="cpu")[0]
clip_model.eval().requires_grad_(False)
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# 512 is embed dimension for ViT-B/16 CLIP
aesthetic_model = AestheticMeanPredictionLinearModel(512)
aesthetic_model_raw_path = os.path.join(os.path.dirname(__file__),
                                        "simulacra_vit_b_16_linear.pth")
aesthetic_model_raw = torch.load(aesthetic_model_raw_path)
aesthetic_model.load_state_dict(aesthetic_model_raw)
aesthetic_model = aesthetic_model.to("cpu")


def judge(img_path):
    img = Image.open(img_path).convert("RGB")
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224,224))
    img = TF.to_tensor(img).to("cpu")
    img = normalize(img)
    latent = F.normalize(clip_model.encode_image(img[None, ...]).float(),
                         dim=-1)
    return aesthetic_model(latent)
