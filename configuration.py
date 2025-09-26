from transformers import PretrainedConfig
from typing import List, Optional, Tuple


class PneumoniaConfig(PretrainedConfig):
    model_type = "pneumonia"

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        feature_dim: int = 256,
        seg_dropout: float = 0.1,
        cls_dropout: float = 0.1,
        seg_num_classes: int = 1,
        cls_num_classes: int = 1,
        in_chans: int = 1,
        img_size: Tuple[int, int] = (512, 512),  # height, width
        decoder_n_blocks: int = 5,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        encoder_channels: List[int] = [24, 48, 64, 160, 256],
        decoder_center_block: bool = False,
        decoder_norm_layer: str = "bn",
        decoder_attention_type: Optional[str] = None,
        **kwargs,
    ):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.seg_dropout = seg_dropout
        self.cls_dropout = cls_dropout
        self.seg_num_classes = seg_num_classes
        self.cls_num_classes = cls_num_classes
        self.in_chans = in_chans
        self.img_size = img_size
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_channels = decoder_channels
        self.encoder_channels = encoder_channels
        self.decoder_center_block = decoder_center_block
        self.decoder_norm_layer = decoder_norm_layer
        self.decoder_attention_type = decoder_attention_type
        super().__init__(**kwargs)
