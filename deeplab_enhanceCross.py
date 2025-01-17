import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import cv2
import numpy as np
from modulesshare import *
# from modulessharebranch2 import *
from utils import *
import Config as config
affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#用于构建一个瓶颈（Bottleneck）结构的残差块。
#瓶颈结构是一种高效的网络设计，它通过减少中间层的通道数来降低计算复杂度，同时保持较高的性能。
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  #change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 这个模块的目的是通过对输入特征应用多个具有不同膨胀率的卷积层，
# 来提取多尺度的特征，并将这些特征融合在一起以进行分类。
class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        # 使用 zip 函数将 dilation_series 和 padding_series 列表中的元素配对，
        # 然后遍历这些配对，为每个配对创建一个卷积层，并将其添加到 self.conv2d_list中
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        vis = True
        config_vit = config.get_CTranS_config()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        #???????
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 创建一个由多个相同类型的残差块组成的层,,dilation=4: 卷积层的扩张率
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        #????????
        self.mtcs1 = ChannelTransformer_share(config_vit, vis, 32, channel_num=[64, 128, 256, 512, 2048], patchSize=config_vit.patch_sizes)
        # self.mtcs2 = ChannelTransformer_single(config_vit, vis, 32, channel_num=[64, 128, 256, 512, 512], patchSize=config_vit.patch_sizes)
        # [6, 12, 18, 24]: 这两个列表可能表示分类模块中不同分支的滤波器尺寸或者其他相关参数。
        self.layer5 = self._make_pred_layer( Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        #权重初始化的过程
            # 遍历模型的所有模块：
            # self.modules() 是一个生成器，它会返回模型中的所有模块（包括子模块）。通过遍历这些模块，可以对它们的参数进行操作。
            # 判断模块类型并进行相应的初始化：
            # isinstance(m, nn.Conv2d) 检查当前模块 m 是否为二维卷积层 (nn.Conv2d)。如果是，那么执行以下操作：
            # 计算 n，它是卷积核的大小乘以输出通道数，代表了每个过滤器的参数数量。
            # 使用 normal_ 方法将卷积层的权重数据初始化为均值为0，标准差为0.01的正态分布。这是一种常用的权重初始化方法，有助于控制网络中信号的尺度，避免梯度消失或爆炸。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        # 一个包含卷积和批量归一化层的序列模块，用于调整维度以匹配主路径的输出
        # 这段代码检查是否需要进行下采样（downsample）。如果步长不为1，
        # 或者输入平面数不等于输出平面数乘以块的扩展系数，或者使用了特定的膨胀率，则需要下采样。
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        #这段代码遍历下采样模块中的批量归一化层的参数，并将它们的requires_grad属性设置为False，
        # 这意味着在训练过程中这些参数不会更新（即冻结这些层）。
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion

        #循环添加剩余的块到layers列表中，但不包含下采样。
        #最后，使用nn.Sequential将所有的层组合成一个模块并返回。
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, y):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer5(x)

        y = self.conv1(y) #64*128*128
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y) #64*64*64
        y = self.layer1(y)  #256*64*64
        y = self.layer2(y)  #512*32*32
        y = self.layer3(y)  #1024*32*32
        y1 = self.layer5(y)

        x2 = self.layer4(x)
        y2 = self.layer4(y)  #2048*32*32

        #### abla: no MBATrans
        # # attx, atty = self.mtcs1(x2, y2)
        # x2 = self.layer6(x2)
        # y2 = self.layer6(y2)
        # # attxp = self.avg_pool(attx)
        # # attyp = self.avg_pool(atty)
    
        # return x1, x2, y1, y2

        ## MBATA-GAN
        attx, atty = self.mtcs1(x2, y2)
        x2 = self.layer6(attx)
        y2 = self.layer6(atty)
        # s = 256
        # x_visualize = F.interpolate(attx, size=(s, s), mode='bilinear', align_corners=False)
        # x_visualize = x_visualize.detach().cpu().numpy() # 用Numpy处理返回的[1,256,513,513]特征图
        # x_visualize = np.mean(x_visualize, axis=1).reshape(s, s)  # shape为[513,513]，二维
        # x_visualize = (
        #             ((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(
        #     np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
        # savedir = './attx2.jpg'
        # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
        # cv2.imwrite(savedir, x_visualize)
        
        # x_visualize = F.interpolate(atty, size=(s, s), mode='bilinear', align_corners=False)
        # x_visualize = x_visualize.detach().cpu().numpy() # 用Numpy处理返回的[1,256,513,513]特征图
        # x_visualize = np.mean(x_visualize, axis=1).reshape(s, s)  # shape为[513,513]，二维
        # x_visualize = (
        #             ((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(
        #     np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
        # savedir = './atty2.jpg'
        # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
        # cv2.imwrite(savedir, x_visualize)
        
        attxp = self.avg_pool(attx)
        attyp = self.avg_pool(atty)
    
        return x1, x2, y1, y2, attx, atty, attxp, attyp

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.mtcs1)
        # b.append(self.mtcs2)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': LEARNING_RATE},
                {'params': self.get_10x_lr_params(), 'lr': 10 * LEARNING_RATE_D}]


def DeeplabMulti(num_classes=21):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

