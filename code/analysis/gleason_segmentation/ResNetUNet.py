# Implementation of UNet model with pre-trained ResNet-18 blocks for convolution
#
# Code based upon PyTorch ResNet-18 UNet by Naoto Usuyama
#
# http://github.com/usuyama/pytorch-unet/
#
# Original source code licensed under MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class, n_channels=3):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        self.base_in_features = [n_channels, 64, 64, 128, 256, 512]
        self.base_out_features = [64, 64, 256, 512, 512, 1024]
        self.model_features = [512, 512, 256, 128, 64, 64, 64]
        
        self.base_layers = list(self.base_model.children())
        if n_channels != 3:
            X = nn.Conv2d(self.level_features[0][0], self.level_features[0][1], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
            self.layer0 = nn.Sequential(*([X]+self.base_layers[1:3])) # size=(N, 64, x.H/2, x.W/2)
        else:
            self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(self.base_in_features[1], self.base_out_features[1], 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(self.base_in_features[2], self.base_out_features[2], 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(self.base_in_features[3], self.base_out_features[3], 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(self.base_in_features[4], self.base_out_features[4], 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(self.base_in_features[5], self.base_out_features[5], 1, 0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(self.base_out_features[4] + self.base_out_features[5], self.model_features[0], 3, 1)
        self.conv_up2 = convrelu(self.base_out_features[3] + self.model_features[0], self.model_features[1], 3, 1)
        self.conv_up1 = convrelu(self.base_out_features[2] + self.model_features[1], self.model_features[2], 3, 1)
        self.conv_up0 = convrelu(self.base_out_features[1] + self.model_features[2], self.model_features[3], 3, 1)
        
        self.conv_original_size0 = convrelu(n_channels, self.model_features[4], 3, 1)
        self.conv_original_size1 = convrelu(self.model_features[4], self.model_features[5], 3, 1)
        self.conv_original_size2 = convrelu(self.model_features[5] + self.model_features[3], self.model_features[6], 3, 1)
        
        self.conv_last = nn.Conv2d(self.model_features[6], n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out
