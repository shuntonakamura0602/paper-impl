from functools import partial
from typing import Any,Callable,Optional,Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int,out_planes: int,stride: int = 1,groups: int = 1,dilation:int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups, # グループ化畳み込みを制御するためのパラメータ
        bias=False, # 直後にバッチ正規化層(nn.BatchNorm2d)を置くため役割が重複する
        dilation=dilation # フィルタの内側に隙間を作る。計算コストを抑えつつ受容野を広げる
    )

def conv1x1(in_planes: int,out_planes: int,stride: int =  1) -> nn.Conv2d:
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

## BasicBlockの実装
class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None, # nn.Moduleを受け取るか、あるいは何も指定がなければNoneになる
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[...,nn.Module]] = None, # 意味: 任意の引数を取り、nn.Moduleのインスタンスを返す関数（やクラス）
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample #ステージ間の移行部分で、次元のつじつまを合わせるために使われる
        self.stride = stride
    
    def forward(self,x: Tensor) ->Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsmaple is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

## Bottleneckの実装
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[...,nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes,width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width,width,stride,groups,dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width,planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.donwnsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: type[Union[BasicBlock,Bottleneck]],
            layers: list[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            gruops: int = 1,
            width_per_group: int = 64,
            reeplace_stride_with_dilation: Optional[list[bool]] = None,
            norm_layer: Optional[Callable[...,nn.Module]] = None,
    ) -> None:
        super().__init__()
        
