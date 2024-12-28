import torch
from torch import nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import cv2 

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import RepDense2DBasicBlock, RepDense2DBasicBlockV, RepVGGBlock
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling_dense
import pdb
import time

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   dilation=1, conv_type='subm', norm_cfg=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, dilation=dilation,
                                 padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None):
    m = spconv.SparseSequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2,eps=1e-3, momentum=0.01)#nn.BatchNorm2d(num_features=in_channels, eps=1e-3, momentum=0.01)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

@BACKBONES.register_module
class RepMiddlePillarEncoder2X34_6632(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, pc_range=[-75.2, -75.2, 75.2, 75.2],
            deploy=False, use_SPD=False, leakyrelu=False, atten_pool=False, use_max=False,
            name="RepMiddlePillarEncoder2X34_6632", **kwargs
    ):
        super(RepMiddlePillarEncoder2X34_6632, self).__init__()
        self.name = name
        self.deploy = deploy
        self.use_SPD = use_SPD
        self.atten_pool = atten_pool
        self.use_max =  use_max
        self.leakyrelu = leakyrelu
        if self.leakyrelu:
            self.act=nn.LeakyReLU()
        else:
            self.act=nn.ReLU()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling_dense(
            mlps=[6 + num_input_features, 64],
            bev_size=pillar_cfg['pool']['bev'], #pool1 #pool2: 0.15
            point_cloud_range=pc_range, leakyrelu=self.leakyrelu, atten_pool=self.atten_pool, use_max=self.use_max,
        )  # [752, 752]

        self.conv2_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_2 = int(128 * 0.5)
        self.conv2 = nn.Sequential(
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv3_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_3 = int(256 * 0.5)
        self.conv3 = nn.Sequential(
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv4_first_block = nn.Sequential(
                RepVGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_4 = 256
        self.conv4 = nn.Sequential(
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv5_first_block = nn.Sequential(
            RepVGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu),)
        c_5 = 512
        self.conv5 = nn.Sequential(
            RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.backbone_channels = {
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 512,
        }
        self.backbone_strides = {
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def _make_layer(self, inplanes, planes, layer_num, stride=1):
        cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ]
        for k in range(layer_num):
            cur_layers.extend([
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
                nn.ReLU()])

        return nn.Sequential(*cur_layers)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv2 = sp_tensor.dense() # [B, 32, 1440, 1440]
        # x_conv2 = sp_tensor # [B, 32, 1440, 1440]
        x_conv2 = self.conv2_first_block(sp_tensor)
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3_first_block(x_conv2)
        x_conv3 = self.conv3(x_conv3)
        x_conv4 = self.conv4_first_block(x_conv3)
        x_conv4 = self.conv4(x_conv4)
        x_conv5 = self.conv5_first_block(x_conv4)
        x_conv5 = self.conv5(x_conv5) # [B, 256, 90, 90]
        return dict(
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class RepMiddlePillarEncoder2X34_6622(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, pc_range=[-75.2, -75.2, 75.2, 75.2],
            deploy=False, use_SPD=False, leakyrelu=False, atten_pool=False, use_max=False,
            name="RepMiddlePillarEncoder2X34_6622", **kwargs
    ):
        super(RepMiddlePillarEncoder2X34_6622, self).__init__()
        self.name = name
        self.deploy = deploy
        self.use_SPD = use_SPD
        self.atten_pool = atten_pool
        self.use_max =  use_max
        self.leakyrelu = leakyrelu
        if self.leakyrelu:
            self.act=nn.LeakyReLU()
        else:
            self.act=nn.ReLU()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling_dense(
            mlps=[6 + num_input_features, 64],
            bev_size=pillar_cfg['pool']['bev'], #pool1 #pool2: 0.15
            point_cloud_range=pc_range, leakyrelu=self.leakyrelu, atten_pool=self.atten_pool, use_max=self.use_max,
        )  # [752, 752]

        self.conv2_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_2 = int(128 * 0.5)
        self.conv2 = nn.Sequential(
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv3_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_3 = int(256 * 0.5)
        self.conv3 = nn.Sequential(
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv4_first_block = nn.Sequential(
                RepVGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_4 = 256
        self.conv4 = nn.Sequential(
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            #RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv5_first_block = nn.Sequential(
            RepVGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu),)
        c_5 = 512
        self.conv5 = nn.Sequential(
            RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.backbone_channels = {
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 512,
        }
        self.backbone_strides = {
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def _make_layer(self, inplanes, planes, layer_num, stride=1):
        cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ]
        for k in range(layer_num):
            cur_layers.extend([
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
                nn.ReLU()])

        return nn.Sequential(*cur_layers)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv2 = sp_tensor.dense() # [B, 32, 1440, 1440]
        # x_conv2 = sp_tensor # [B, 32, 1440, 1440]
        x_conv2 = self.conv2_first_block(sp_tensor)
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3_first_block(x_conv2)
        x_conv3 = self.conv3(x_conv3)
        x_conv4 = self.conv4_first_block(x_conv3)
        x_conv4 = self.conv4(x_conv4)
        x_conv5 = self.conv5_first_block(x_conv4)
        x_conv5 = self.conv5(x_conv5) # [B, 256, 90, 90]
        return dict(
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class RepMiddlePillarEncoder2X34_6631(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, pc_range=[-75.2, -75.2, 75.2, 75.2],
            deploy=False, use_SPD=False, leakyrelu=False, atten_pool=False, use_max=False,
            name="RepMiddlePillarEncoder2X34_6631", **kwargs
    ):
        super(RepMiddlePillarEncoder2X34_6631, self).__init__()
        self.name = name
        self.deploy = deploy
        self.use_SPD = use_SPD
        self.atten_pool = atten_pool
        self.use_max =  use_max
        self.leakyrelu = leakyrelu
        if self.leakyrelu:
            self.act=nn.LeakyReLU()
        else:
            self.act=nn.ReLU()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling_dense(
            mlps=[6 + num_input_features, 64],
            bev_size=pillar_cfg['pool']['bev'], #pool1 #pool2: 0.15
            point_cloud_range=pc_range, leakyrelu=self.leakyrelu, atten_pool=self.atten_pool, use_max=self.use_max,
        )  # [752, 752]

        self.conv2_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_2 = 64
        self.conv2 = nn.Sequential(
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_2, c_2, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv3_first_block=nn.Sequential(
                RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_3 = 128
        self.conv3 = nn.Sequential(
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_3, c_3, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv4_first_block = nn.Sequential(
                RepVGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu))
        c_4 = 256
        self.conv4 = nn.Sequential(
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(c_4, c_4, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )
        self.conv5_first_block = nn.Sequential(
            RepVGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, deploy=self.deploy, leakyrelu=self.leakyrelu),)
        c_5 = 512
        self.conv5 = nn.Sequential(
            RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
            #RepDense2DBasicBlock(c_5, c_5, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.backbone_channels = {
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 512,
        }
        self.backbone_strides = {
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def _make_layer(self, inplanes, planes, layer_num, stride=1):
        cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ]
        for k in range(layer_num):
            cur_layers.extend([
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
                nn.ReLU()])

        return nn.Sequential(*cur_layers)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv2 = sp_tensor.dense() # [B, 32, 1440, 1440]
        # x_conv2 = sp_tensor # [B, 32, 1440, 1440]
        x_conv2 = self.conv2_first_block(sp_tensor)
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3_first_block(x_conv2)
        x_conv3 = self.conv3(x_conv3)
        x_conv4 = self.conv4_first_block(x_conv3)
        x_conv4 = self.conv4(x_conv4)
        x_conv5 = self.conv5_first_block(x_conv4)
        x_conv5 = self.conv5(x_conv5) # [B, 256, 90, 90]
        return dict(
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class RepMiddlePillarEncoder2X34_big_6611(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, pc_range=[-75.2, -75.2, 75.2, 75.2],
            deploy=False, use_SPD=False,  leakyrelu=False, atten_pool=False, use_max=False, export_onnx = False,
            name="RepMiddlePillarEncoder2X34_big_6611", **kwargs
    ):
        super(RepMiddlePillarEncoder2X34_big_6611, self).__init__()
        self.name = name
        self.deploy = deploy
        self.use_SPD = use_SPD
        self.atten_pool = atten_pool
        self.use_max =  use_max
        self.leakyrelu = leakyrelu
        self.export_onnx = export_onnx
        if self.leakyrelu:
            self.act=nn.LeakyReLU()
        else:
            self.act=nn.ReLU()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPooling_dense(
            mlps=[6 + num_input_features, 64],
            bev_size=pillar_cfg['pool']['bev'], #pool1 #pool2: nus: 0.15m waymo:0.2m
            point_cloud_range=pc_range, leakyrelu=self.leakyrelu, atten_pool=self.atten_pool, use_max=self.use_max,
        )  # [752, 752]

        self.conv2 = nn.Sequential(
            RepDense2DBasicBlockV(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(64, 64, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.conv3 = nn.Sequential(
            RepVGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, deploy=self.deploy),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
            RepDense2DBasicBlock(128, 128, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.conv4 = nn.Sequential(
            RepVGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, deploy=self.deploy),
            RepDense2DBasicBlock(256, 256, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        # norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            RepVGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, deploy=self.deploy),
            RepDense2DBasicBlock(512, 512, deploy=self.deploy, leakyrelu=self.leakyrelu),
        )

        self.backbone_channels = {
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 512,
        }
        self.backbone_strides = {
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def _make_layer(self, inplanes, planes, layer_num, stride=1):
        cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ]
        for k in range(layer_num):
            cur_layers.extend([
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
                nn.ReLU()])

        return nn.Sequential(*cur_layers)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv2 = sp_tensor
        # sp_tensor = sp_tensor.dense()
        # print('current backbone is  RepMiddlePillarEncoder2X34')
        x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )