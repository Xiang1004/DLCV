
from torch import nn
from transformers import ViTForImageClassification

class ViT(nn.Module):

    def __init__(self):
        super().__init__()
        self.feats = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.linear = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 37),
        )
        
    def forward(self, x):
        x = self.feats(x)
        x = x.logits
        x = self.linear(x)
        return x

