# Implementation of multi-resolution parallel UNet model
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

from ResNetUNet import ResNetUNet; 
import torch
import torch.nn as nn

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

class ResNetUNetEnsemble(nn.Module):

    def __init__(self, n_class, n_models):
        super().__init__()
        
        self.n_class = n_class
        self.n_models = n_models
        
        self.base_models = nn.ModuleList([ResNetUNet(n_class, 3) for i in range(self.n_models)])
        
        self.base_in_features = [sum(x) for x in zip(*[model.base_in_features for model in self.base_models])]
        self.base_out_features = [sum(x) for x in zip(*[model.base_out_features for model in self.base_models])]
        self.model_features = [512, 512, 256, 128, 64, sum([model.model_features[5] for model in self.base_models]), 64]
        
        self.layer0_1x1 = convrelu(self.base_in_features[1], self.base_out_features[1], 1, 0)
        self.layer1_1x1 = convrelu(self.base_in_features[2], self.base_out_features[2], 1, 0)
        self.layer2_1x1 = convrelu(self.base_in_features[3], self.base_out_features[3], 1, 0)
        self.layer3_1x1 = convrelu(self.base_in_features[4], self.base_out_features[4], 1, 0)
        self.layer4_1x1 = convrelu(self.base_in_features[5], self.base_out_features[5], 1, 0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(self.base_out_features[4] + self.base_out_features[5], self.model_features[0], 3, 1)
        self.conv_up2 = convrelu(self.base_out_features[3] + self.model_features[0], self.model_features[1], 3, 1)
        self.conv_up1 = convrelu(self.base_out_features[2] + self.model_features[1], self.model_features[2], 3, 1)
        self.conv_up0 = convrelu(self.base_out_features[1] + self.model_features[2], self.model_features[3], 3, 1)
        
        self.conv_original_size2 = convrelu(self.model_features[5] + self.model_features[3], self.model_features[6], 3, 1)
        
        self.conv_last = nn.Conv2d(self.model_features[6], n_class, 1)
        
    def forward(self, input):
        x_original = [self.base_models[i].conv_original_size0(input[:,(3*i):(3*(i+1)),:,:]) for i in range(self.n_models)]
        x_original = [self.base_models[i].conv_original_size1(x_original[i]) for i in range(self.n_models)]
        x_original = torch.cat(x_original, dim=1)
        
        layer0 = [self.base_models[i].layer0(input[:,(3*i):(3*(i+1)),:,:]) for i in range(self.n_models)]
        layer1 = [self.base_models[i].layer1(layer0[i]) for i in range(self.n_models)]
        layer2 = [self.base_models[i].layer2(layer1[i]) for i in range(self.n_models)]
        layer3 = [self.base_models[i].layer3(layer2[i]) for i in range(self.n_models)]
        layer4 = [self.base_models[i].layer4(layer3[i]) for i in range(self.n_models)]

        layer4 = self.layer4_1x1(torch.cat(layer4, dim=1))
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(torch.cat(layer3, dim=1))
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(torch.cat(layer2, dim=1))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(torch.cat(layer1, dim=1))
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(torch.cat(layer0, dim=1))
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out
