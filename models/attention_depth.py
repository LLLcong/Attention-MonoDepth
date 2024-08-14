import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings
from .swin_transformer import SwinTransformer


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, input_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
        self.conv_transpose = nn.ConvTranspose2d(input_features, input_features, 2, stride=2)

    def forward(self, x, concat_with):
        up_x = self.conv_transpose(x)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], num_classes = 256):
        super(DecoderBN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels[3], in_channels[3], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels[3]), nn.LeakyReLU())

        self.up1 = UpSampleBN(skip_input=in_channels[3]+in_channels[2], output_features=in_channels[2], input_features=in_channels[3])
        self.up2 = UpSampleBN(skip_input=in_channels[2]+in_channels[1], output_features=in_channels[1], input_features=in_channels[2])
        self.up3 = UpSampleBN(skip_input=in_channels[1]+in_channels[0], output_features=in_channels[0], input_features=in_channels[1])
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels[0], in_channels[0],  2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[1]),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels[1], in_channels[1], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[1]),
                                  nn.LeakyReLU())
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels[0], in_channels[0], 2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[0]),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_channels[0]),
                                  nn.LeakyReLU(),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels[0], num_classes, kernel_size=1, stride=1, padding=0))
        self.activate =  nn.Softmax(dim=1)

        self.regressor = nn.Sequential(nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256))

    def forward(self, features):
        x_block1, x_block2, x_block3, x_block4 = features

        x_d0 = self.conv1(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)

        x_d3 = self.conv_transpose1(x_d3)
        x_d4 = self.conv2(x_d3)
        return x_d0, x_d1, x_d2, x_d3, x_d4


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)



def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


# Attentional feature fusion block
class AFF(nn.Module):
    def __init__(self, channels=96, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            # F.adaptive_avg_pool2d(feature_map, (1,1))
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# Bin generation module
class BinWidthGeneration_AFF(nn.Module):
    def __init__(self, min_depth=1e-3, max_depth=16, in_features=96, hidden_features=256*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # self.aff1 = AFF(channels=384)
        # self.aff2 = AFF(channels=192)
        self.aff3 = AFF(channels=96)
        # self.conv1 = nn.Conv2d(768, 384, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, e_list): 
        # f2_up = upsample(e_list[2], 2)
        # f2_conv = self.conv2(f2_up)
        # f3 = self.aff2(f2_conv, e_list[3])
        f3_up = upsample(e_list[2], 2)
        f3_conv = self.conv3(f3_up)
        f3_conv_up = upsample(f3_conv, 2)
        f4 = self.aff3(f3_conv_up, e_list[3])
        f4_up = upsample(f4, 2)

        bcp0_out = torch.mean(f4_up.flatten(start_dim=2), dim = 2)
        bcp0_out = self.fc1(bcp0_out)
        bcp0_out = self.act(bcp0_out)
        bcp0_out = self.drop(bcp0_out)
        bcp0_out = self.fc2(bcp0_out)
        bcp0_out = self.drop(bcp0_out)

        return bcp0_out



class AttentionDepth(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, pretrained=None):
        super(AttentionDepth, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val

        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        in_channels = [96, 192, 384, 768]
        window_size = 7
        frozen_stages = -1

        backbone_swin = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )
        self.backbone = SwinTransformer(**backbone_swin)

        self.decoder = DecoderBN(in_channels=in_channels)
        self.conv_out = nn.Sequential(nn.Conv2d(192, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

        self.binwidth_aff = BinWidthGeneration_AFF(min_depth=self.min_val, max_depth=self.max_val, in_features=in_channels[0], out_features=self.num_classes)
        self.init_weights(pretrained=pretrained)    

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)


    def forward(self, x, **kwargs):
        enc = self.backbone(x)
        d0, d1, d2, d3, unet_out = self.decoder(enc, **kwargs)
        out = self.conv_out(unet_out)

        bcp0_out = self.binwidth_aff([d0, d1, d2, d3])

        bins = torch.softmax(bcp0_out, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_val - self.min_val) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1) 

        pred = torch.sum(out * centers, dim=1, keepdim=True) 
        return bin_edges, pred


    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.backbone.parameters()  

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.conv_out, self.binwidth_aff]   
        for m in modules:
            yield from m.parameters()

