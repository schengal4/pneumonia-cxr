import albumentations as A
import torch
import torch.nn as nn

from numpy.typing import NDArray
from transformers import PreTrainedModel
from timm import create_model
from typing import Optional
from .configuration import PneumoniaConfig
from .unet import UnetDecoder, SegmentationHead

_PYDICOM_AVAILABLE = False
try:
    from pydicom import dcmread
    from pydicom.pixels import apply_voi_lut

    _PYDICOM_AVAILABLE = True
except ModuleNotFoundError:
    pass


class PneumoniaModel(PreTrainedModel):
    config_class = PneumoniaConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = create_model(
            model_name=config.backbone,
            features_only=True,
            pretrained=False,
            in_chans=config.in_chans,
        )
        self.decoder = UnetDecoder(
            decoder_n_blocks=config.decoder_n_blocks,
            decoder_channels=config.decoder_channels,
            encoder_channels=config.encoder_channels,
            decoder_center_block=config.decoder_center_block,
            decoder_norm_layer=config.decoder_norm_layer,
            decoder_attention_type=config.decoder_attention_type,
        )
        self.img_size = config.img_size
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.seg_num_classes,
            size=self.img_size,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=config.cls_dropout)
        self.classifier = nn.Linear(config.feature_dim, config.cls_num_classes)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 255] -> [-1, 1]
        mini, maxi = 0.0, 255.0
        x = (x - mini) / (maxi - mini)
        x = (x - 0.5) * 2.0
        return x

    @staticmethod
    def load_image_from_dicom(path: str) -> Optional[NDArray]:
        if not _PYDICOM_AVAILABLE:
            print("`pydicom` is not installed, returning None ...")
            return None
        dicom = dcmread(path)
        arr = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            # invert image if needed
            arr = arr.max() - arr

        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = (arr * 255).astype("uint8")
        return arr

    def preprocess(self, x: NDArray) -> NDArray:
        x = A.Resize(self.img_size[0], self.img_size[1], p=1)(image=x)["image"]
        return x

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.normalize(x)
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        logits = self.segmentation_head(decoder_output[-1])
        b, n = features[-1].shape[:2]
        features = self.pooling(features[-1]).reshape(b, n)
        features = self.dropout(features)
        cls_logits = self.classifier(features)
        out = {
            "mask": logits,
            "cls": cls_logits
        }
        if return_logits:
            return out
        out["mask"] = out["mask"].sigmoid()
        out["cls"] = out["cls"].sigmoid()
        return out
