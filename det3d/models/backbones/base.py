import torch
import numpy as np
from torch import nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
except:
    import spconv
    from spconv import SparseConv3d, SubMConv3d

from timm.models.layers import DropPath
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels,eps=1e-3, momentum=0.01))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, leakyrelu=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        # print('deploy: ', self.deploy)
        self.groups = groups
        self.in_channels = in_channels
        self.leakyrelu = leakyrelu
        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        if self.leakyrelu:
            self.nonlinearity = nn.LeakyReLU()
        else:
            self.nonlinearity = nn.ReLU()

        if use_se: # False in train
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy: # False in train
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels, eps=1e-3, momentum=0.01) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs, relu=True):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if relu:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            return self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            print(branch)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        print('switch rep to deploy')
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1 * 4, c2, k, s, autopad(k, p), groups=g, bias=False),
                                nn.BatchNorm2d(c2),
                                nn.SiLU())

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
    else:
        return spconv.SparseConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )

def conv2D1x1(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            # indice_key=indice_key,
        )


class Sparse2DBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlock, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlockV, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = replace_feature(x, self.relu(x.features))
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class RepDense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        deploy=False, 
        use_se=False,
        use_SPD=False,
        leakyrelu=False,
    ):
        super(RepDense2DBasicBlock, self).__init__()
        self.deploy = deploy
        self.use_se = use_se
        self.leakyrelu = leakyrelu
        self.conv1 = RepVGGBlock(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy=self.deploy, use_se=self.use_se, leakyrelu=self.leakyrelu)
        self.conv2 = RepVGGBlock(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy=self.deploy, use_se=self.use_se, leakyrelu=self.leakyrelu)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        return out

class RepDense2DBasicBlockV(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        deploy=False, 
        use_se=False,
        use_SPD=False,
        leakyrelu=False,
    ):
        super(RepDense2DBasicBlockV, self).__init__()
        self.deploy = deploy
        self.use_se = use_se
        self.use_SPD = use_SPD
        self.leakyrelu = leakyrelu
        self.conv0 = RepVGGBlock(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy=self.deploy, use_se=self.use_se, leakyrelu=self.leakyrelu)
        self.conv1 = RepVGGBlock(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se, leakyrelu=self.leakyrelu)
        self.conv2 = RepVGGBlock(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se, leakyrelu=self.leakyrelu)

    def forward(self, x):
        x = self.conv0(x)
        out = self.conv1(x)
        out = self.conv2(out)

        return out
