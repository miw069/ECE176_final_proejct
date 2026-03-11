import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, z_dim: int = 256, pretrained: bool = True):
        super().__init__()
        base = resnet18(weights="DEFAULT" if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Linear(512, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)


class ImplicitOccDecoder(nn.Module):
    def __init__(self, z_dim: int = 256, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = 3 + z_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        z_exp = z.unsqueeze(1).expand(B, N, z.shape[-1])
        x = torch.cat([points, z_exp], dim=-1)
        return self.net(x).squeeze(-1)


class PointCloudEncoder(nn.Module):
    def __init__(self, z_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp1 = nn.Linear(3, hidden)
        self.mlp2 = nn.Linear(hidden, hidden)
        self.mlp3 = nn.Linear(hidden, z_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.mlp1(pts))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x.max(dim=1).values


class ImplicitOccNet(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        pretrained_encoder: bool = True,
        decoder_hidden: int = 256,
        decoder_depth: int = 4,
        use_cad_encoder: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.image_encoder = ImageEncoder(z_dim=z_dim, pretrained=pretrained_encoder)
        self.occ_decoder = ImplicitOccDecoder(z_dim=z_dim, hidden=decoder_hidden, depth=decoder_depth)
        self.cad_encoder = PointCloudEncoder(z_dim=z_dim) if use_cad_encoder else None

    def forward(self, image: torch.Tensor, points: torch.Tensor):
        z = self.image_encoder(image)
        logits = self.occ_decoder(points, z)
        return logits, z

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_cad(self, cad_points: torch.Tensor) -> torch.Tensor:
        if self.cad_encoder is None:
            raise RuntimeError("cad_encoder disabled; set use_cad_encoder=True")
        return self.cad_encoder(cad_points)