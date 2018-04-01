import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        '''
        f中的每个元素的size分别是 bs 256 w/4 h/4， bs 512 w/8 h/8， 
        bs 1024 w/16 h/16， bs 2048 w/32 h/32
        '''
        return x, f

def resnet50(pretrained, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print (('==> loading pretrained state dict from %s' % pretrained))
        model.load_state_dict(torch.load(pretrained))
        print ("==> loading pretrained state dict done")

    return model

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images: bs * w * h * channel 
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:,i,:,:] -= means[i]

    return images

class East(nn.Module):
    def __init__(self, pretrain):
        super(East, self).__init__()
        self.resnet = resnet50(pretrain)
        self.conv1x1_1 = nn.Conv2d(3072, 128, 1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.relu1 = nn.ReLU()

        self.conv3x3_1 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU()

        self.conv1x1_2 = nn.Conv2d(640, 64, 1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.relu3 = nn.ReLU()

        self.conv3x3_2 = nn.Conv2d(64, 64, 3 ,padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()

        self.conv1x1_3 = nn.Conv2d(320, 64, 1)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.relu5 = nn.ReLU()

        self.conv3x3_3 = nn.Conv2d(64, 32, 3, padding=1)
        # self.bn6 = nn.BatchNorm2d(32)
        # self.relu6 = nn.ReLU()

        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        # self.bn7 = nn.BatchNorm2d(32)
        # self.relu7 = nn.ReLU()

        self.conv_score = nn.Conv2d(32, 1, 1)

        self.conv_geo = nn.Conv2d(32, 4, 1)

        self.conv_angle = nn.Conv2d(32, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,images):
        images = mean_image_subtraction(images)
        _, f = self.resnet(images)

        g = [None, None, None, None]
        h = [None, None, None, None]

        h[3] = f[3]
        g[3] = self.unpool(h[3])

        h[2] = self.conv3x3_1(self.conv1x1_1(torch.cat((g[3], f[2]), dim=1)))
        g[2] = self.unpool(h[2])

        h[1] = self.conv3x3_2(self.conv1x1_2(torch.cat((g[2], f[1]), dim=1)))
        g[1] = self.unpool(h[1])

        h[0] = self.conv3x3_3(self.conv1x1_3(torch.cat((g[1], f[0]), dim=1)))
        g[0] = self.conv(h[0])

        F_score = self.conv_score(g[0]) #  bs 1 w/4 h/4
        F_score = self.sigmoid(F_score)

        geo_map = self.conv_geo(g[0])
        geo_map = self.sigmoid(geo_map) * 512

        angle_map = self.conv_angle(g[0])
        angle_map = self.sigmoid(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        F_geometry = torch.cat((geo_map, angle_map), 1) # bs 5 w/4 w/4
        return F_score, F_geometry
