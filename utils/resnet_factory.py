from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, low_dim)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, representation=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if representation:
            return out
        # out = self.l2norm(out)
        return self.fc(out)

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)
        # hidden_dim = int(out_dim/2)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x



class SimSiamWithCls(nn.Module):

    '''
    SimSiam with Classifier
    '''
    def __init__(self, arch='resnet18', feat_dim=2048, num_proj_layers=2):
        
        super(SimSiamWithCls, self).__init__()
        self.backbone = models.resnet18()
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(out_dim, 10)

        pred_hidden_dim = int(feat_dim / 4)

        self.projector = nn.Sequential(
            nn.Linear(out_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, feat_dim),
        )

    def forward(self, img, im_aug1=None, im_aug2=None):

        r_ori = self.backbone(img)
        if im_aug1 is None and im_aug2 is None:
            cls = self.classifier(r_ori)
            rep = self.projector(r_ori)
            return {'cls': cls, 'rep':rep}
        else:

            r1 = self.backbone(im_aug1)
            r2 = self.backbone(im_aug2)

            z1 = self.projector(r1)
            z2 = self.projector(r2)
            # print("shape of z:", z1.shape)

            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            # print("shape of p:", p1.shape)

            return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}

class ByolWithCls(nn.Module):

    '''
    SimSiam with Classifier
    '''
    def __init__(self, base_encoder=models.resnet18(), proj_hidden_dim =4096, proj_output_dim=256):
        super(ByolWithCls, self).__init__()
        self.backbone = base_encoder
        self.features_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.features_dim, 10)
        

        # 3-layer projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )


    def forward(self, img, representation=False, cls=False):

        r_ori = self.backbone(img)
        if representation:
            return r_ori
        ori_out = self.classifier(r_ori)
        if cls:
            return ori_out
        ssl_embedding  = self.projector(r_ori)
        return ssl_embedding

class SwavWithCls(nn.Module):

    '''
    SimSiam with Classifier
    '''
    def __init__(self, base_encoder=models.resnet18(), proj_hidden_dim =2048, proj_output_dim=128):
        super(SwavWithCls, self).__init__()
        self.backbone = base_encoder
        self.features_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.features_dim, 10)
        

        # 3-layer projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )


    def forward(self, img, representation=False, cls=False):

        r_ori = self.backbone(img)
        if representation:
            return r_ori
        ori_out = self.classifier(r_ori)
        if cls:
            return ori_out
        ssl_embedding  = self.projector(r_ori)
        return ssl_embedding

class Mocov3WithCls(nn.Module):

    '''
    SimSiam with Classifier
    '''
    def __init__(self, base_encoder=models.resnet18(), proj_hidden_dim =4096, proj_output_dim=256):
        super(Mocov3WithCls, self).__init__()
        self.backbone = base_encoder
        self.features_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.features_dim, 10)
        

        # 3-layer projector
        self.projector = self._build_mlp(
                2,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
            )


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=True))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    def forward(self, img, representation=False, cls=False):

        r_ori = self.backbone(img)
        if representation:
            return r_ori
        ori_out = self.classifier(r_ori)
        if cls:
            return ori_out
        ssl_embedding  = self.projector(r_ori)
        return ssl_embedding

class DeepclusterV2WithCls(nn.Module):

    '''
    SimSiam with Classifier
    '''
    def __init__(self, base_encoder=models.resnet18(), proj_hidden_dim =2048, proj_output_dim=128):
        super(DeepclusterV2WithCls, self).__init__()
        self.backbone = base_encoder
        self.features_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.features_dim, 10)
        
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )



    def forward(self, img, representation=False, cls=False):

        r_ori = self.backbone(img)
        if representation:
            return r_ori
        ori_out = self.classifier(r_ori)
        if cls:
            return ori_out
        ssl_embedding  = self.projector(r_ori)
        return ssl_embedding
    
class RN18_ssl_branch(nn.Module):

    def __init__(self, out_dim, feat_dim=2048, num_proj_layers=2) -> None:
        super(RN18_ssl_branch, self).__init__()
        self.projector = projection_MLP(out_dim, feat_dim, num_proj_layers)
        self.predictor = prediction_MLP(feat_dim)
    
    def forward(self, im_aug1, im_aug2):
        
        z1 = self.projector(im_aug1)
        z2 = self.projector(im_aug2)
        # print("shape of z:", z1.shape)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}



def ResNet18(low_dim=10):
    return ResNet(BasicBlock, [2,2,2,2], low_dim)

def ResNet34(low_dim=10):
    return ResNet(BasicBlock, [3,4,6,3], low_dim)

def ResNet50(low_dim=10):
    return ResNet(Bottleneck, [3,4,6,3], low_dim)

def ResNet101(low_dim=10):
    return ResNet(Bottleneck, [3,4,23,3], low_dim)

def ResNet152(low_dim=10):
    return ResNet(Bottleneck, [3,8,36,3], low_dim)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
