"""
BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation
Implementation tailored for Drishti-Kavach Railway Track & Rail Line Perception.

Reference:
  Yu et al., "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation", IJCV 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """
    Detail Branch: High spatial resolution, shallow depth (1/8 downsampling, 128 channels).
    Captures thin rail line edges and fine ballast textures.
    """
    def __init__(self):
        super(DetailBranch, self).__init__()
        # Stage 1: 1/2 downsample
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        # Stage 2: 1/4 downsample
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        # Stage 3: 1/8 downsample
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, ks=3, stride=2, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        feat = self.s1(x)
        feat = self.s2(feat)
        feat = self.s3(feat)
        return feat


class StemBlock(nn.Module):
    """
    Stem Block of Semantic Branch (1/4 downsample).
    """
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv_in = ConvBNReLU(3, 16, ks=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, ks=1, stride=1, padding=0),
            ConvBNReLU(8, 16, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv_in(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat_cat = torch.cat([feat_left, feat_right], dim=1)
        return self.fuse(feat_cat)


class GELayer(nn.Module):
    """
    Gather-and-Expansion (GE) Layer for Semantic Branch.
    """
    def __init__(self, in_chan, out_chan, stride=1, exp_ratio=6):
        super(GELayer, self).__init__()
        self.stride = stride
        mid_chan = in_chan * exp_ratio

        if stride == 1:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1),
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1),
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 1:
            return self.relu(self.conv(x) + x)
        else:
            return self.relu(self.conv(x) + self.shortcut(x))


class CEBlock(nn.Module):
    """
    Context Embedding Block (Global contextual reasoning).
    """
    def __init__(self, in_chan=128, out_chan=128):
        super(CEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_chan)
        self.conv_gap = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = self.gap(x)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        return self.conv_last(x + feat)


class SemanticBranch(nn.Module):
    """
    Semantic Branch: Rapid downsampling (1/32 scale) for wide contextual awareness.
    """
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.s12 = StemBlock()                       # 1/4, 16 ch
        self.s3 = nn.Sequential(
            GELayer(16, 32, stride=2),               # 1/8, 32 ch
            GELayer(32, 32, stride=1),
        )
        self.s4 = nn.Sequential(
            GELayer(32, 64, stride=2),               # 1/16, 64 ch
            GELayer(64, 64, stride=1),
        )
        self.s5 = nn.Sequential(
            GELayer(64, 128, stride=2),              # 1/32, 128 ch
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            CEBlock(128, 128),
        )

    def forward(self, x):
        feat2 = self.s12(x)
        feat3 = self.s3(feat2)
        feat4 = self.s4(feat3)
        feat5 = self.s5(feat4)
        return feat2, feat3, feat4, feat5


class BGALayer(nn.Module):
    """
    Bilateral Guided Aggregation (BGA) Layer.
    Fuses high-res detail cues with deep semantic features.
    """
    def __init__(self, detail_chan=128, sem_chan=128, out_chan=128):
        super(BGALayer, self).__init__()
        # Detail path
        self.detail_dw = nn.Sequential(
            nn.Conv2d(detail_chan, detail_chan, kernel_size=3, stride=1, padding=1, groups=detail_chan, bias=False),
            nn.BatchNorm2d(detail_chan),
            nn.Conv2d(detail_chan, detail_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.detail_down = nn.Sequential(
            ConvBNReLU(detail_chan, detail_chan, ks=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

        # Semantic path
        self.sem_dw = nn.Sequential(
            ConvBNReLU(sem_chan, sem_chan, ks=3, stride=1, padding=1),
            nn.Conv2d(sem_chan, sem_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.sem_up = ConvBNReLU(sem_chan, sem_chan, ks=3, stride=1, padding=1)

        self.conv_out = ConvBNReLU(detail_chan, out_chan, ks=3, stride=1, padding=1)

    def forward(self, feat_d, feat_s):
        # Detail branch outputs 1/8 scale
        # Semantic branch s5 outputs 1/32 scale -> upsample to 1/8
        d_size = feat_d.size()[2:]

        # Path 1: Detail guided by Semantic
        d_feat = self.detail_dw(feat_d)
        s_feat_up = F.interpolate(feat_s, size=d_size, mode='bilinear', align_corners=False)
        s_feat_gate = self.sem_dw(s_feat_up)
        path1 = d_feat * s_feat_gate

        # Path 2: Semantic guided by Detail
        d_feat_down = self.detail_down(feat_d)
        d_feat_gate = torch.sigmoid(d_feat_down)
        s_feat = self.sem_up(feat_s)
        s_feat_interp = F.interpolate(s_feat, size=d_feat_down.size()[2:], mode='bilinear', align_corners=False)
        path2 = s_feat_interp * d_feat_gate
        path2_up = F.interpolate(path2, size=d_size, mode='bilinear', align_corners=False)

        return self.conv_out(path1 + path2_up)


class SegmentHead(nn.Module):
    """
    Final Segmentation Prediction Head.
    """
    def __init__(self, in_chan, mid_chan, num_classes, up_factor=8):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, num_classes, kernel_size=1, stride=1, padding=0)
        self.up_factor = up_factor

    def forward(self, x, target_size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        logits = self.conv_out(feat)
        if target_size is not None:
            return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        elif self.up_factor > 1:
            return F.interpolate(logits, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        return logits


class BiSeNetV2(nn.Module):
    """
    Complete BiSeNetV2 Architecture for Railway Perception (Drishti-Kavach).
    Outputs 3 classes: 0 (Background), 1 (Track_Bed), 2 (Rail_Lines).
    """
    def __init__(self, num_classes=3, is_training=True):
        super(BiSeNetV2, self).__init__()
        self.is_training = is_training
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BGALayer(detail_chan=128, sem_chan=128, out_chan=128)
        self.head = SegmentHead(128, 1024, num_classes, up_factor=8)

        # Auxiliary Booster Heads (Active during training for deep supervision)
        if is_training:
            self.aux2 = SegmentHead(16, 64, num_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, num_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 256, num_classes, up_factor=16)
            self.aux5 = SegmentHead(128, 512, num_classes, up_factor=32)

    def forward(self, x):
        img_size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5 = self.segment(x)
        feat_fuse = self.bga(feat_d, feat5)

        out = self.head(feat_fuse, target_size=img_size)

        if self.is_training and self.training:
            out_aux2 = self.aux2(feat2, target_size=img_size)
            out_aux3 = self.aux3(feat3, target_size=img_size)
            out_aux4 = self.aux4(feat4, target_size=img_size)
            out_aux5 = self.aux5(feat5, target_size=img_size)
            return out, out_aux2, out_aux3, out_aux4, out_aux5

        return out


if __name__ == "__main__":
    # Sanity check
    model = BiSeNetV2(num_classes=3, is_training=False)
    x = torch.randn(2, 3, 512, 1024)
    out = model(x)
    print("Model forward pass successful!")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {params:,} (~{params/1e6:.2f}M)")
