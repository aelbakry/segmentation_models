import os
import sys
from functools import partial

import torch.hub
from torch.nn import Dropout2d, UpsamplingBilinear2d, Sequential
from torch.nn.functional import upsample_bilinear
from torch.utils import model_zoo

import resnet
from densenet import densenet121, densenet169, densenet161
from dpn import dpn92, dpn131, dpn107, dpn92_mc
from resnet import resnext50_32x4d, resnext101_32x8d
from senet import se_resnext50_32x4d, se_resnext101_32x4d, senet154, SCSEModule, SCSEScaledModule
from nfnet import dm_nfnet_f0, dm_nfnet_f1, dm_nfnet_f2, dm_nfnet_f3, dm_nfnet_f4, dm_nfnet_f5, dm_nfnet_f6
from cbam import CBAM
from layers import ScaledStdConv2dSame, GammaAct

from timm import create_model

encoder_params = {

 'dm_nfnet_f0':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f0,
         'url': None},

    'dm_nfnet_f1':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f1,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth'},

    'dm_nfnet_f2':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f2,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f2-89875923.pth'},

    'dm_nfnet_f3':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f3,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f3-d74ab3aa.pth'},

    'dm_nfnet_f4':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f4,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f4-0ac5b10b.pth'},
    'dm_nfnet_f5':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f5,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f5-ecb20ab1.pth'},
    'dm_nfnet_f6':
        {'filters': [128, 256, 512, 1536, 1536],
         'decoder_filters': [64, 128, 256, 384],
         'init_op': dm_nfnet_f6,
         'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f6-e0f12116.pth'},


    'seresnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 192, 256],
         'init_op': se_resnext50_32x4d,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'},
    'senet154':
        {'filters': [128, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'init_op': senet154,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'},
    'seresnext50_fat':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [96, 192, 256, 512],
         'last_upsample': 64,
         'init_op': se_resnext50_32x4d,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'},
    'seresnext101':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': se_resnext101_32x4d,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'},
    'dpn92':
        {'filters': [64, 336, 704, 1552, 2688],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': dpn92,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'dpn92_mc':
        {'filters': [64, 336, 704, 1552, 2688],
         'decoder_filters': [48, 96, 256, 256],
         'last_upsample': 48,
         'init_op': partial(dpn92_mc, num_channels=3),
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'resnet34':
        {'filters': [64, 64, 128, 256, 512],
         'decoder_filters': [64, 128, 256, 512],
         'last_upsample': 64,
         'init_op': resnet.resnet34,
         'url': resnet.model_urls['resnet34']},
    'resnet101':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [48, 96, 256, 256],
         'last_upsample': 48,
         'init_op': resnet.resnet101,
         'url': resnet.model_urls['resnet101']},
    'resnext101':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 192, 256, 256],
         'last_upsample': 64,
         'init_op': resnext101_32x8d,
          #using weights from WSL
         'url': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'},
    'resnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': resnext50_32x4d,
         'url': "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"},
    'resnext50_3':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(resnext50_32x4d, in_channels=3),
         'url': "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"},
    'dpn131':
        {'filters': [128, 352, 832, 1984, 2688],
         'init_op': dpn131,
         'last_upsample': 64,
         'decoder_filters': [64, 128, 256, 256],
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn131-71dfe43e0.pth'},
    'dpn107':
        {'filters': [128, 376, 1152, 2432, 2688],
         'init_op': dpn107,
         'last_upsample': 64,
         'decoder_filters': [64, 128, 256, 256],
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-1ac7121e2.pth'},
    'resnet50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': resnet.resnet50(),
         'url': resnet.model_urls['resnet50']},
    'densenet121':
        {'filters': [64, 256, 512, 1024, 1024],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'url': None,
         'init_op': densenet121},
    'densenet169':
        {'filters': [64, 256, 512, 1280, 1664],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'url': None,
         'init_op': densenet169},
    'densenet161':
        {'filters': [96, 384, 768, 2112, 2208],
         'decoder_filters': [48, 96, 256, 256],
         'last_upsample': 48,
         'url': None,
         'init_op': densenet161},
    'densenet161_3':
        {'filters': [96, 384, 768, 2112, 2208],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'url': None,
         'init_op': partial(densenet161, in_channels=3)},
    'resnet152d':
         {'filters': [64, 256, 512, 1024, 2048],
          'decoder_filters': [64, 128, 256, 384],
          'last_upsample': 64,
          'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
          'init_op': resnet.resnet_152d},
    'resnet200d':
      {'filters': [64, 256, 512, 1024, 2048],
       'decoder_filters': [64, 128, 256, 384],
       'last_upsample': 64,
       'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
       'init_op': resnet.resnet_200d}

}

import torch
from torch import nn
import torch.nn.functional as F


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvReLu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.ReLU)


class ConvReLu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.ReLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[self.first_layer_params_names[0] + '.weight' ].data
            #init RGB channels for post disaster image as well
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, 3:6, ...] = pretrained_dict[self.first_layer_params_names[0] + '.weight' ].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34', scaled=False):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block') and scaled:
            self.decoder_block = UnetDecoderScaledBlock
        else:
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        if not hasattr(self, 'use_bilinear_4x'):
            self.use_bilinear_4x = False

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])


        if self.first_layer_stride_two:
            # self.last_upsample = self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
            #                                         self.last_upsample_filters)
            # TODO: make it configurable
            self.last_upsample = UpsamplingBilinear2d(scale_factor=2)
        if self.use_bilinear_4x:
            self.final = self.make_final_classifier(self.decoder_filters[1], num_classes)
        else:
            self.final = self.make_final_classifier(
                self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0], num_classes)
        self._initialize_weights()
        self.dropout = Dropout2d(p=0.25)
        encoder = encoder_params[encoder_name]['init_op']()
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None and self.pretrained:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []

        for stage in self.encoder_stages:
            #            x = self.dropout(x)
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        bottlenecks = self.bottlenecks
        if self.use_bilinear_4x:
            bottlenecks = bottlenecks[:-1]

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        #if self.use_bilinear_4x:
        #x = self.dropout(x)

        if not self.use_bilinear_4x and self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        if self.use_bilinear_4x:
            f = upsample_bilinear(f, scale_factor=4)

        return f

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        if self.attention:
            return nn.Sequential(CBAM(in_filters),
                nn.Conv2d(in_filters, num_classes, 1, padding=0)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_filters, num_classes, 1, padding=0)
            )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class ConvSCSEBottleneckScaled(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, mode='concat', attention=False):
        print("bottleneck ", in_channels, out_channels)
        super().__init__()

        self.seq = nn.Sequential(
            ScaledStdConv2dSame(in_channels, out_channels, kernel_size=3),
            GammaAct(),
            SCSEModule(out_channels, reduction=reduction, mode=mode)
        )

    def forward(self, dec, enc):

        if dec.shape[-1] != enc.shape[-1]:
            enc = F.interpolate(enc, scale_factor=2, mode='bilinear')

        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class UnetDecoderScaledBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ScaledStdConv2dSame(in_channels, out_channels, kernel_size=3),
            GammaAct()
        )

    def forward(self, x):
        return self.layer(x)

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class SCSeResnetD(EncoderDecoder):

    def __init__(self, seg_classes, backbone_arch, reduction=2, mode='concat', num_channels=3, pretrained=True, attention=False, scaled=False):
        if not hasattr(self, 'bottleneck_type') and scaled:
            self.bottleneck_type = partial(ConvSCSEBottleneckScaled, reduction=reduction, mode=mode, attention=attention)
        else:
            self.bottleneck_type = partial(ConvSCSEBottleneckNoBn, reduction=reduction, mode=mode, attention=attention)
        self.first_layer_stride_two = True
        self.concat_scse = mode == 'concat'
        self.pretrained = pretrained
        self.attention = attention

        super().__init__(seg_classes,num_channels , backbone_arch)
        self.last_upsample = self.decoder_block(
            self.decoder_filters[0] * 2 if self.concat_scse else self.decoder_filters[0],
            self.last_upsample_filters,
            self.last_upsample_filters)

    def calc_dec_filters(self, d_filters):
        return d_filters * 2 if self.concat_scse else d_filters

    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())
        dec_results = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        mask = self.final(x)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        if self.concat_scse and layer + 1 < len(self.decoder_filters):
            in_channels *= 2

        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.act1)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['layer0.conv1']

class SCSeResneXt(EncoderDecoder):

    def __init__(self, seg_classes, backbone_arch, reduction=2, mode='concat', num_channels=3, pretrained=True, attention=False):
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = partial(ConvSCSEBottleneckNoBn, reduction=reduction, mode=mode, attention=attention)
        self.first_layer_stride_two = True
        self.concat_scse = mode == 'concat'
        self.pretrained = pretraineds
        self.attention = attention

        super().__init__(seg_classes,num_channels , backbone_arch)
        self.last_upsample = self.decoder_block(
            self.decoder_filters[0] * 2 if self.concat_scse else self.decoder_filters[0],
            self.last_upsample_filters,
            self.last_upsample_filters)

    def calc_dec_filters(self, d_filters):
        return d_filters * 2 if self.concat_scse else d_filters

    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())
        dec_results = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        mask = self.final(x)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        if self.concat_scse and layer + 1 < len(self.decoder_filters):
            in_channels *= 2

        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['layer0.conv1']

class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, mode='concat', attention=False):
        print("bottleneck ", in_channels, out_channels)
        super().__init__()
        if attention:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                SCSEModule(out_channels, reduction=reduction, mode=mode),
                CBAM(out_channels*2)
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                SCSEModule(out_channels, reduction=reduction, mode=mode)
            )

    def forward(self, dec, enc):

        if dec.shape[-1] != enc.shape[-1]:
            enc = F.interpolate(enc, scale_factor=2, mode='bilinear')

        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class ResnetD(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch, pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.act1)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

class Resnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch, pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ResneXt(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch,  num_channels=3, pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, num_channels, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['conv1']


class DPNUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='dpn92', num_channels=3, pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, num_channels, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        elif layer == 1:
            return nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    @property
    def first_layer_params_names(self):
        return ['features.conv1_1.conv']

class DensenetUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='dpn92', pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.features.conv0,  # conv
                encoder.features.norm0,  # bn
                encoder.features.relu0  # relu
            )
        elif layer == 1:
            return nn.Sequential(encoder.features.pool0, encoder.features.denseblock1)
        elif layer == 2:
            return nn.Sequential(encoder.features.transition1, encoder.features.denseblock2)
        elif layer == 3:
            return nn.Sequential(encoder.features.transition2, encoder.features.denseblock3)
        elif layer == 4:
            return nn.Sequential(encoder.features.transition3, encoder.features.denseblock4, encoder.features.norm5,
                                 nn.ReLU())


class SEUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='senet154', pretrained=True, attention=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

class SCSeNfNet(EncoderDecoder):

    def __init__(self, seg_classes, backbone_arch, reduction=2, mode='concat', num_channels=3, pretrained=True, attention=False, scaled=False):
        if not hasattr(self, 'bottleneck_type') and scaled:
            self.bottleneck_type = partial(ConvSCSEBottleneckScaled, reduction=reduction, mode=mode, attention=attention)
        else:
            self.bottleneck_type = partial(ConvSCSEBottleneckNoBn, reduction=reduction, mode=mode, attention=attention)

        self.first_layer_stride_two = True
        self.concat_scse = mode == 'concat'
        self.pretrained = pretrained
        self.attention = attention

        super().__init__(seg_classes,num_channels , backbone_arch, scaled)
        self.last_upsample = self.decoder_block(
            self.decoder_filters[0] * 2 if self.concat_scse else self.decoder_filters[0],
            self.last_upsample_filters,
            self.last_upsample_filters)

    def calc_dec_filters(self, d_filters):
        return d_filters * 2 if self.concat_scse else d_filters

    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())
        dec_results = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        mask = self.final(x)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        if self.concat_scse and layer + 1 < len(self.decoder_filters):
            in_channels *= 2

        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.stem
        elif layer == 1:
            return encoder.stages[0]
        elif layer == 2:
            return encoder.stages[1]
        elif layer == 3:
            return encoder.stages[2]
        elif layer == 4:
            return encoder.stages[3]

    @property
    def first_layer_params_names(self):
        return ['stem.conv1']

class NfNet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='senet154', pretrained=True, attention=False, scaled=False):
        self.first_layer_stride_two = True
        self.pretrained = pretrained
        self.attention = attention

        super().__init__(seg_classes, 3, backbone_arch, scaled)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.stem
        elif layer == 1:
            return encoder.stages[0]
        elif layer == 2:
            return encoder.stages[1]
        elif layer == 3:
            return encoder.stages[2]
        elif layer == 4:
            return encoder.stages[3]




setattr(sys.modules[__name__], 'scse_unet', partial(SCSeResneXt))
setattr(sys.modules[__name__], 'se_unet', partial(SEUnet))
setattr(sys.modules[__name__], 'scse_unet_addition', partial(SCSeResneXt, reduction=16, mode='addition'))
setattr(sys.modules[__name__], 'resnet_unet', partial(Resnet))
setattr(sys.modules[__name__], 'resnext_unet', partial(ResneXt))
setattr(sys.modules[__name__], 'resnext_unet_3', partial(ResneXt, num_channels=3))
setattr(sys.modules[__name__], 'dpn_unet', partial(DPNUnet))
setattr(sys.modules[__name__], 'densenet_unet', partial(DensenetUnet))

__all__ = ['scse_unet',
           'scse_unet_addition',
           'resnet_unet',
           'resnext_unet',
           'resnext_unet_3',
           'dpn_unet',
           'se_unet',
           'densenet_unet',
           'irv_unet',
           ]


if __name__ == '__main__':
    import numpy as np

    d = SCSeNfNet(1, backbone_arch="dm_nfnet_f0", pretrained=False, attention=False, scaled=True)
    d.eval()
    with torch.no_grad():
        images = torch.from_numpy(np.zeros((1, 3, 256, 256), dtype="float32"))
        i = d(images)
    print(d)
    print(i.size())
