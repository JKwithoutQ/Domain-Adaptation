import torch.nn as nn
from functions import ReverseLayerF, GradientReverseLayer, WarmStartGradientReverseLayer
import torch
import resnet

class DANN_Resnet(nn.Module):
    """
    DANN model
    为什么不同大小的DANN能够影响最终的结果！！！？？？
    """
    def __init__(self, n_class=10):
        self.n_class = n_class
        super(DANN_Resnet, self).__init__()
        # feature
        self.backbone = resnet.resnet50(pretrained=True)
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        # classifer
        self.classifer = nn.Sequential(
            nn.Linear(1024, self.n_class)
        )
        # discriminator
        # self.grl = GradientReverseLayer()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.discriminator = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.discriminator1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # feature
        x = self.backbone(input_data)  # x.shape = [batch, 2048, 7, 7]
        x = self.bottleneck(x)
        # classifer
        classifer = self.classifer(x)
        # discriminator
        domain = self.grl(x)
        domain = self.discriminator(domain)
        return classifer, domain

