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

        if self.downsample is not None:
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
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初期値を設定
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # バッチ正規化層の重みを0にすると、その層の出力は常に0になる。
        # これにより、学習開始時点では、残差ブロックの分岐経路からの出力が0になり、ブロック全体が恒等写像のように振る舞う。
        # この安全な初期状態たら学習を始めることで深いネットワークでも勾配が不安定になるのを防ぎ、学習の序盤をスムーズに進めることができる。
        """

        # このブロックは、ResNetの学習を安定させるためのテクニック「Zero-Init-Residual」を実装しています。
        #

        # --- 新入社員の研修に例えた解説 ---
        # ショートカット接続(identity)は、指導役が教える「安全な既存のやり方」。
        # 残差ブロックの分岐経路(畳み込み層)は、新入社員が試す「新しいクリエイティブなアイデア」。
        # 最後のバッチ正規化(BN)層は、その新しいアイデアの「承認ゲート」の役割を果たします。

        # この研修方針(zero_init_residual)が有効な場合、以下の処理を行います。
        if zero_init_residual:
            # 会社(モデル)にいる全社員(全モジュール)を一人ずつチェックします。
            for m in self.modules():
                # もし社員が「Bottleneck」タイプで、その最終承認ゲート(bn3)が存在すれば...
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    # 承認ゲートを最初は固く閉じておきます（重みを0に初期化）。
                    nn.init.constant_(m.bn3.weight, 0)
                # もし社員が「BasicBlock」タイプで、その最終承認ゲート(bn2)が存在すれば...
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    # こちらも同様に、承認ゲートを閉じておきます（重みを0に初期化）。
                    nn.init.constant_(m.bn2.weight, 0)

        # === この処理の効果 ===
        #
        # 承認ゲートを閉じる（重みを0にする）ことで、学習開始時点では「新しいアイデア」の経路からの出力は0になります。
        # その結果、各残差ブロックは out = identity + 0 となり、入力をそのまま返す恒等写像(identity)のように振る舞います。
        # 
        # これにより、モデルはまず「安全な既存のやり方」だけを学習することに集中できます。
        # そして、学習が進むにつれて、本当に役立つ「新しいアイデア」が見つかった場合にのみ、
        # 徐々に承認ゲートを開けて（重みを0から更新して）、そのアイデアを取り入れていきます。
        #
        # この堅実なアプローチにより、非常に深いネットワークでも学習が安定し、最終的な精度が向上します。
        """
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
        
