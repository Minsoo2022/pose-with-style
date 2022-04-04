import math
import random
import functools
import operator
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class SpatiallyModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.gamma = nn.Sequential(*[EqualConv2d(in_channel, 128, kernel_size=1), nn.ReLU(True), EqualConv2d(128, in_channel, kernel_size=1)])
        self.beta = nn.Sequential(*[EqualConv2d(in_channel, 128, kernel_size=1), nn.ReLU(True), EqualConv2d(128, in_channel, kernel_size=1)])

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def modulate(self, x, gamma, beta):
        return gamma * x + beta

    def normalize(self, x):
        mean, std = self.calc_mean_std(x)
        mean = mean.expand_as(x)
        std = std.expand_as(x)
        return (x-mean)/std

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.scale * self.weight.squeeze(0)

        gamma = self.gamma(style)
        beta = self.beta(style)

        input = self.modulate(input, gamma, beta)

        if self.upsample:
            weight = weight.transpose(0, 1)
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2
            )
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)
        else:
            out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

        out = self.normalize(out)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        spatial=False,
    ):
        super().__init__()

        if spatial:
            self.conv = SpatiallyModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                upsample=upsample,
                blur_kernel=blur_kernel,
            )
        else:
            self.conv = ModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
            )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], spatial=False):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        if spatial:
            self.conv = SpatiallyModulatedConv2d(in_channel, 3, 1)
        else:
            self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class PoseEncoder(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256):
        super().__init__()
        self.size = size
        convs = [ConvLayer(3, ngf, 1)]
        convs.append(ResBlock(ngf, ngf*2, blur_kernel))
        convs.append(ResBlock(ngf*2, ngf*4, blur_kernel))
        convs.append(ResBlock(ngf*4, ngf*8, blur_kernel))
        convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        if self.size == 512:
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        if self.size == 1024:
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out

class InputEncoder(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256):
        super().__init__()
        self.size = size
        input_ch = 3
        convs = [ConvLayer(input_ch, ngf, 1)]

        if size == 128:
            convs.append(ResBlock(ngf, ngf * 2, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 2, ngf * 4, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 4, ngf * 8, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 8, ngf * 8, blur_kernel, downsample=False))
        elif size == 256:
            convs.append(ResBlock(ngf, ngf * 2, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 2, ngf * 4, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 4, ngf * 8, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 8, ngf * 8, blur_kernel, downsample=True))
        elif size == 512:
            convs.append(ResBlock(ngf, ngf * 2, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 2, ngf * 4, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 4 + 128, ngf * 8, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 8, ngf * 8, blur_kernel, downsample=True))
            convs.append(ResBlock(ngf * 8, ngf * 8, blur_kernel, downsample=True))
        # else:
        #     raise NotImplementedError()
        # if self.size == 512:
        #     convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        # if self.size == 1024:
        #     convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        #     convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))

        self.convs = nn.Sequential(*convs)

    def forward(self, input_img, input_feature):
        for i, layer in enumerate(self.convs):
            if i == 0:
                out = layer(input_img)
            elif i == 3:
                out = layer(torch.cat((out, input_feature), dim=1))
            else:
                out = layer(out)
        return out


class SpatialAppearanceEncoder(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256):
        super().__init__()
        self.size = size
        self.dp_uv_lookup_256_np = np.load('util/dp_uv_lookup_256.npy')
        input_nc = 4 # source RGB and sil
        input_condition_nc = 3

        self.conv1 = ConvLayer(input_nc, ngf, 1)                #  ngf 256 256
        self.conv2 = ResBlock(ngf, ngf*2, blur_kernel)          # 2ngf 128 128
        self.conv3 = ResBlock(ngf*2, ngf*4, blur_kernel)        # 4ngf 64  64
        self.conv4 = ResBlock(ngf*4, ngf*8, blur_kernel)        # 8ngf 32  32
        self.conv5 = ResBlock(ngf*8, ngf*8, blur_kernel)        # 8ngf 16  16
        if self.size == 512:
            self.conv6 = ResBlock(ngf*8, ngf*8, blur_kernel)    # 8ngf 16  16 - starting from ngf 512 512
        if self.size == 1024:
            self.conv6 = ResBlock(ngf*8, ngf*8, blur_kernel)
            self.conv7 = ResBlock(ngf*8, ngf*8, blur_kernel)    # 8ngf 16  16 - starting from ngf 1024 0124

        # self.conv11 = EqualConv2d(ngf+ngf+1, ngf*8, 1)
        # self.conv21 = EqualConv2d(ngf*2+ngf*2+1, ngf*8, 1)
        # self.conv31 = EqualConv2d(ngf*4+ngf*4+1, ngf*8, 1)
        # self.conv41 = EqualConv2d(ngf*8+ngf*4+1, ngf*8, 1)
        # self.conv51 = EqualConv2d(ngf*8+ngf*4+1, ngf*8, 1)
        if self.size == 512:
            self.conv61 = EqualConv2d(ngf*8+ngf*4, ngf*8, 1)
        if self.size == 1024:
            self.conv61 = EqualConv2d(ngf*8+1, ngf*8, 1)
            self.conv71 = EqualConv2d(ngf*8+1, ngf*8, 1)

        # if self.size == 1024:
        #     self.conv13 = EqualConv2d(ngf*8, int(ngf/2), 3, padding=1)
        #     self.conv23 = EqualConv2d(ngf*8, ngf*1, 3, padding=1)
        #     self.conv33 = EqualConv2d(ngf*8, ngf*2, 3, padding=1)
        #     self.conv43 = EqualConv2d(ngf*8, ngf*4, 3, padding=1)
        #     self.conv53 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)
        #     self.conv63 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)
        # elif self.size == 512:
        #     self.conv13 = EqualConv2d(ngf*8, ngf*1, 3, padding=1)
        #     self.conv23 = EqualConv2d(ngf*8, ngf*2, 3, padding=1)
        #     self.conv33 = EqualConv2d(ngf*8, ngf*4, 3, padding=1)
        #     self.conv43 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)
        #     self.conv53 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)
        # else:
        #     self.conv13 = EqualConv2d(ngf*8, ngf*2, 3, padding=1)
        #     self.conv23 = EqualConv2d(ngf*8, ngf*4, 3, padding=1)
        #     self.conv33 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)
        #     self.conv43 = EqualConv2d(ngf*8, ngf*8, 3, padding=1)

        self.conconv1 = ConvLayer(input_condition_nc, ngf, 1)            # 128 128
        self.conconv2 = ResBlock(ngf, ngf*2, blur_kernel)          # 64 64
        self.conconv3 = ResBlock(ngf*2, ngf*4, blur_kernel)        # 32 32
        self.conconv4 = ResBlock(ngf*4, ngf*4, blur_kernel)        # 16 16
        self.conconv5 = ResBlock(ngf * 4, ngf * 4, blur_kernel)
        self.conconv6 = ResBlock(ngf * 4, ngf * 4, blur_kernel)


        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, flow, sil, condition):
        condition = torch.nn.functional.interpolate(condition, size=(self.size, self.size), mode='bilinear', align_corners=True)

        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        if self.size == 512:
            x6 = self.conv6(x5)
        if self.size == 1024:
            x6 = self.conv6(x5)
            x7 = self.conv7(x6)

        # warp- get flow
        # todo mask 어떻게 할지
        # pose_mask = 1-(pose[:,0, :, :] == 0).float().unsqueeze(1)
        # warp- resize flow
        # f1 = torch.nn.functional.interpolate(flow, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        # f2 = torch.nn.functional.interpolate(flow, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=True)
        # f3 = torch.nn.functional.interpolate(flow, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=True)
        # f4 = torch.nn.functional.interpolate(flow, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=True)
        # f5 = torch.nn.functional.interpolate(flow, size=(x5.shape[2], x5.shape[3]), mode='bilinear', align_corners=True)
        if self.size == 512:
            f6 = torch.nn.functional.interpolate(flow, size=(x6.shape[2], x6.shape[3]), mode='bilinear', align_corners=True)
        if self.size == 1024:
            f6 = torch.nn.functional.interpolate(flow, size=(x6.shape[2], x6.shape[3]), mode='bilinear', align_corners=True)
            f7 = torch.nn.functional.interpolate(flow, size=(x7.shape[2], x7.shape[3]), mode='bilinear', align_corners=True)
        # warp- now warp
        # x1 = torch.nn.functional.grid_sample(x1, f1.permute(0,2,3,1))
        # x2 = torch.nn.functional.grid_sample(x2, f2.permute(0,2,3,1))
        # x3 = torch.nn.functional.grid_sample(x3, f3.permute(0,2,3,1))
        # x4 = torch.nn.functional.grid_sample(x4, f4.permute(0,2,3,1))
        # x5 = torch.nn.functional.grid_sample(x5, f5.permute(0,2,3,1))
        if self.size == 512:
            x6 = torch.nn.functional.grid_sample(x6, f6.permute(0,2,3,1))
        if self.size == 1024:
            x6 = torch.nn.functional.grid_sample(x6, f6.permute(0,2,3,1))
            x7 = torch.nn.functional.grid_sample(x7, f7.permute(0,2,3,1))


        v1 = self.conconv1(condition)
        v2 = self.conconv2(v1)
        v3 = self.conconv3(v2)
        v4 = self.conconv4(v3)
        v5 = self.conconv5(v4)
        v6 = self.conconv6(v5)

        # fpn
        if self.size == 1024:
            F7 = self.conv71(torch.cat([x7,p7], 1))
            f6 = self.up(F7)+self.conv61(torch.cat([x6,p6], 1))
            F6 = self.conv63(f6)
            f5 = self.up(F6)+self.conv51(torch.cat([x5,p5], 1))
            F5 = self.conv53(f5)
            f4 = self.up(f5)+self.conv41(torch.cat([x4,p4], 1))
            F4 = self.conv43(f4)
            f3 = self.up(f4)+self.conv31(torch.cat([x3,p3], 1))
            F3 = self.conv33(f3)
            f2 = self.up(f3)+self.conv21(torch.cat([x2,p2], 1))
            F2 = self.conv23(f2)
            f1 = self.up(f2)+self.conv11(torch.cat([x1,p1], 1))
            F1 = self.conv13(f1)
        elif self.size == 512:
            F6 = self.conv61(torch.cat([x6,v6], 1))
            # f5 = self.up(F6)+self.conv51(torch.cat([x5,p5,v5], 1))
            # F5 = self.conv53(f5)
            # f4 = self.up(f5)+self.conv41(torch.cat([x4,p4,v4], 1))
            # F4 = self.conv43(f4)
            # f3 = self.up(f4)+self.conv31(torch.cat([x3,p3,v3], 1))
            # F3 = self.conv33(f3)
            # f2 = self.up(f3)+self.conv21(torch.cat([x2,p2,v2], 1))
            # F2 = self.conv23(f2)
            # f1 = self.up(f2)+self.conv11(torch.cat([x1,p1,v1], 1))
            # F1 = self.conv13(f1)
        else:
            F5 = self.conv51(torch.cat([x5,p5], 1))
            f4 = self.up(F5)+self.conv41(torch.cat([x4,p4], 1))
            F4 = self.conv43(f4)
            f3 = self.up(f4)+self.conv31(torch.cat([x3,p3], 1))
            F3 = self.conv33(f3)
            f2 = self.up(f3)+self.conv21(torch.cat([x2,p2], 1))
            F2 = self.conv23(f2)
            f1 = self.up(f2)+self.conv11(torch.cat([x1,p1], 1))
            F1 = self.conv13(f1)

        if self.size == 1024:
            return [F7, F6, F5, F4, F3, F2, F1]
        elif self.size == 512:
            return [F6]
        else:
            return [F5, F4, F3, F2, F1]


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        vol_feat_res=32,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        garment_transfer=False,
        part='upper_body',
    ):
        super().__init__()

        self.garment_transfer = garment_transfer
        self.size = size
        self.style_dim = style_dim



        self.appearance_encoder = SpatialAppearanceEncoder(size=size)
        # self.pose_encoder = PoseEncoder(size=size)
        self.input_encoder = InputEncoder(size=512)
        self.style_encoder = ResBlock(512, style_dim, blur_kernel, downsample=True)

        # StyleGAN
        self.channels = {
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.conv1 = StyledConv(
            self.channels[16], self.channels[16], 3, style_dim, blur_kernel=blur_kernel, spatial=False
        )
        self.to_rgb1 = ToRGB(self.channels[16], style_dim, upsample=False, spatial=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 4) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[16]

        for i in range(5, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    spatial=False,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, spatial=False,
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, spatial=False))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2


    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        flow,
        appearance,
        sil,
        input_img,
        input_feat,
        condition,
        styles=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):


        styles = self.appearance_encoder(appearance, flow, sil, condition)
        encoded_style = self.style_encoder(styles[0])
        encoded_style = F.max_pool2d(encoded_style, 8).squeeze(-1).squeeze(-1)

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]


        out = self.input_encoder(input_img, input_feat)
        out = self.conv1(out, encoded_style, noise=noise[0])
        skip = self.to_rgb1(out, encoded_style)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb  in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs#, self.to_silhouettes
        ):
            out = conv1(out, encoded_style, noise=noise1)
            out = conv2(out, encoded_style, noise=noise2)
            skip = to_rgb(out, encoded_style, skip)
            i += 2
        image = skip

        return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.size = size
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(6, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, condition):
        condition = torch.nn.functional.interpolate(condition, size=(self.size, self.size), mode='bilinear', align_corners=True)
        input = torch.cat([input, condition], 1)
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class Discriminator_sub(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], vol_feat_res=128):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.vol_feat_res = vol_feat_res
        in_channel = channels[vol_feat_res]
        vol_convs = [ConvLayer(32, channels[vol_feat_res], 1)]
        for i in range(int(math.log(vol_feat_res, 2)), 5, -1):
            out_channel = channels[2 ** (i -1)]
            vol_convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.vol_convs = nn.Sequential(*vol_convs)

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1): # 9 8 7 6 5 4 3
            out_channel = channels[2 ** (i - 1)]
            if i == 5:
                ch_vol_feat = 512
            else:
                ch_vol_feat = 0
            self.convs.append(ResBlock(in_channel + ch_vol_feat, out_channel, blur_kernel))

            in_channel = out_channel


    def forward(self, x, condition):
        condition = self.vol_convs(condition)
        for i, layer in enumerate(self.convs):
            if i == 0:
                out = layer(x)
            else:
                if out.size(2) == 32:
                    out = layer(torch.cat([out, condition],dim=1))
                else:
                    out = layer(out)
        return out

class Discriminator_feat(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], vol_feat_res=128):
        super().__init__()

        self.sub_block = Discriminator_sub(size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], vol_feat_res=vol_feat_res)

        self.stddev_group = 4
        self.stddev_feat = 1

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.final_conv = ConvLayer(channels[8] + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, condition):
        out = self.sub_block(input, condition)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
