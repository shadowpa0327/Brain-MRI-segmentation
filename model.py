import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import get_config
from networks.vision_transformer import SwinUnet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
# [Start]=================== Unet ==================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        if out_channels == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Sigmoid())
        else:
            # Output logits only if not binary segmentation
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, output_activation=None):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)        
        self.up3 = Up(256, 128//factor, bilinear)        
        self.up4 = Up(128, 64, bilinear)        
        self.outc = OutConv(64, n_classes)
        self.output_act = nn.Sigmoid()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.output_act:
            logits = self.output_act(logits)
        return logits

# [END]=================== Unet ==================

"""
    Build model according to given arguments.
    Parameters:
        input_channels: Number of channels of input image. Default is 3. If grayscale, channel number equals 1.
        num_classes: Number of classes for segmentation. Default is 1 for binary segmentation.
                     Note that if classes > 1, the output of models will be raw logits (no sigmoid).
"""
def build_model(args = None, seg_struct = 'Unet', encoder = 'resnet50', decoder_channels = None ,is_swin = False, is_tran=True, input_channels=3, num_classes=1):
    # Parameters definition
    if num_classes == 1:
        output_activation = 'sigmoid'
    else:
        output_activation = None
    
    if not is_swin and not is_tran:
        encoder_weights = 'imagenet' if args.use_pretrained else None
        if seg_struct == 'Unet':
            if not isinstance(decoder_channels, list):
                raise ValueError("decoder channel of Unet should be a list")
            print(f"Build Unet with encoder {encoder}, pretrained weight:{encoder_weights}")
            if encoder == 'Unet': # using Unet original structure
                model = Unet(n_channels=input_channels,n_classes=num_classes, output_activation=output_activation)
            else : # change Unet encoder
                model = smp.Unet(encoder,in_channels=input_channels, 
                                        encoder_weights=encoder_weights,
                                                classes=num_classes, 
                                            activation=output_activation, 
                                        encoder_depth=5, 
                                    decoder_channels=decoder_channels)
            
        elif seg_struct == 'Unet++':
            if not isinstance(decoder_channels, list):
                raise ValueError("decoder channel of Unet++ should be a list")
            print(f"Build Unet++ with encoder {encoder}, pretrained weight:{encoder_weights}")
            model = smp.UnetPlusPlus(encoder,in_channels=input_channels, 
                                    encoder_weights=encoder_weights,
                                            classes=num_classes, 
                                        activation=output_activation, 
                                    encoder_depth=5, 
                                decoder_channels=decoder_channels)

        elif seg_struct == 'DeepLabV3Plus':
            if not isinstance(decoder_channels, int):
                raise ValueError("decoder channel of Unet++ should be a integer")
            print(f"Build DeepLabV3Plus with encoder {encoder}")
            model = smp.DeepLabV3Plus(encoder,in_channels=input_channels, 
                                    encoder_weights=encoder_weights,
                                            classes=num_classes, 
                                        activation=output_activation, 
                                    encoder_depth=5)
        else : 
            raise ValueError(f"Illegal segmentation structure name {seg_struct}")
        return model
    elif is_swin: # using swin transformer
        config = get_config(args=args)
        model = SwinUnet(config=config, img_size=224, num_classes=1).to('cuda')
        if args.use_pretrained:
            model.load_from(config)
        print(f"Creating model Swin-Unet, using pretrained:{args.use_pretrained}")
        return model
    elif is_tran :
        #config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        #print(config_vit)
        config_vit.n_classes = num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        if args.use_pretrained:
            model.load_from(weights=np.load(args.TransUnet_pretrained_path))
        print(f"Creating model TransUnet, using pretrained:{args.use_pretrained}")
        return model

            
