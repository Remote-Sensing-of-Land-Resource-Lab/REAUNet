import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x0 = self.conv0(x)
        else:
            x0 = x
        x1 = self.conv1(x)
        return F.relu(x0 + x1)


class AttBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super(AttBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels//r, bias=False),
            nn.GELU(),
            nn.Linear(in_channels//r, in_channels, bias=False)
        )
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        channel_avg = self.avg_pool(x).view(b, c)
        channel_avg = self.conv_channel(channel_avg).view(b, c, 1, 1)
        channel_max = self.max_pool(x).view(b, c)
        channel_max = self.conv_channel(channel_max).view(b, c, 1, 1)
        channel = self.sigmoid(channel_avg + channel_max).expand_as(x)

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial = self.sigmoid(self.conv_spatial(spatial)).expand_as(x)

        return x + x * channel + x * spatial


class MPSP(nn.Module):
    def __init__(self, inc, outc):
        super(MPSP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = int(x.shape[-1])
        x1 = F.avg_pool2d(input=x, kernel_size=size)
        y1 = F.interpolate(input=self.conv1(x1), size=size, mode='bilinear', align_corners=False)
        x2 = F.avg_pool2d(input=x, kernel_size=size//2)
        y2 = F.interpolate(input=self.conv2(x2), size=size, mode='bilinear', align_corners=False)
        x3 = F.avg_pool2d(input=x, kernel_size=size//4)
        y3 = F.interpolate(input=self.conv3(x3), size=size, mode='bilinear', align_corners=False)
        x4 = F.avg_pool2d(input=x, kernel_size=size//8)
        y4 = F.interpolate(input=self.conv4(x4), size=size, mode='bilinear', align_corners=False)
        y = torch.cat((y1, y2, y3, y4), dim=1)
        return y


class DownScale(nn.Module):
    def __init__(self, inc, kernel_size=3, stride=2, padding=1, outc=None):
        super(DownScale, self).__init__()

        if not outc:
            outc = inc

        self.down = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outc)
        )

    def forward(self, x):
        return self.down(x)


class EdgeDetect(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.sobel1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=0, bias=False)
        self.sobel2 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=0, bias=False)
        self.sobel3 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=0, bias=False)
        self.sobel4 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=0, bias=False)

        sobel1_kernel = torch.FloatTensor([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]])
        sobel2_kernel = torch.FloatTensor([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]])
        sobel3_kernel = torch.FloatTensor([[-2, -1, 0],
                                           [-1, 0, 1],
                                           [0, 1, 2]])
        sobel4_kernel = torch.FloatTensor([[0, -1, -2],
                                           [1, 0, -1],
                                           [2, 1, 0]])

        # print(in_c, self.sobel1.weight.shape)
        self.sobel1.weight.data = sobel1_kernel.view(1, 1, 3, 3).repeat(in_c, in_c, 1, 1)
        self.sobel2.weight.data = sobel2_kernel.view(1, 1, 3, 3).repeat(in_c, in_c, 1, 1)
        self.sobel3.weight.data = sobel3_kernel.view(1, 1, 3, 3).repeat(in_c, in_c, 1, 1)
        self.sobel4.weight.data = sobel4_kernel.view(1, 1, 3, 3).repeat(in_c, in_c, 1, 1)
        # print(self.sobel1.weight.data)

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_c * 3, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU())

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = F.interpolate(input=x, size=(h+2, w+2), mode='bilinear', align_corners=False)

        gradient1 = self.sobel1(x1)
        gradient2 = self.sobel2(x1)
        out1 = torch.sqrt(gradient1 ** 2 + gradient2 ** 2)
        out1_max_value = torch.max(out1.view(b, c, -1), dim=2)[0]
        out1_max_value = out1_max_value.view(b, c, 1, 1).repeat(1, 1, h, w)
        out1 = out1 / out1_max_value

        gradient3 = self.sobel3(x1)
        gradient4 = self.sobel4(x1)
        out2 = torch.sqrt(gradient3 ** 2 + gradient4 ** 2)
        out2_max_value = torch.max(out2.view(b, c, -1), dim=2)[0]
        out2_max_value = out2_max_value.view(b, c, 1, 1).repeat(1, 1, h, w)
        out2 = out2 / out2_max_value

        out1 = out1.detach()
        out2 = out2.detach()

        x1 = torch.cat([x, out1, out2], dim=1)
        y = self.conv_out(x1)
        return y


class RefUnet(nn.Module):
    def __init__(self,in_channels=1, mid_channels=64):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

        self.conv1 = nn.Conv2d(mid_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, in_channels, 3, padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)
        return x + residual


class REAUNet(nn.Module):
    def __init__(self, in_channels=3, num_class=1):
        super().__init__()

        # ----- Encoder -----
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder1 = nn.Sequential(
            BasicBlock(64, 64),
            EdgeDetect(64),
            BasicBlock(64, 64),
            AttBlock(64)
        )
        self.encoder2 = nn.Sequential(
            DownScale(64),
            BasicBlock(64, 128),
            BasicBlock(128, 128),
            EdgeDetect(128),
            BasicBlock(128, 128),
            AttBlock(128)
        )
        self.encoder3 = nn.Sequential(
            DownScale(128),
            BasicBlock(128, 256),
            BasicBlock(256, 256, kernel_size=5, padding=2),
            EdgeDetect(256),
            BasicBlock(256, 256, kernel_size=5, padding=2),
            BasicBlock(256, 256),
            AttBlock(256)
        )
        self.encoder4 = nn.Sequential(
            DownScale(256),
            BasicBlock(256, 512),
            BasicBlock(512, 512, kernel_size=5, padding=2),
            EdgeDetect(512),
            BasicBlock(512, 512, kernel_size=5, padding=2),
            BasicBlock(512, 512),
            AttBlock(512)
        )
        self.encoder5 = nn.Sequential(
            DownScale(512),
            BasicBlock(512, 512),
            BasicBlock(512, 512, kernel_size=5, padding=2),
            BasicBlock(512, 512),
        )

        # ----- Bridge -----
        self.encoder6 = BasicBlock(512, 1024)
        self.psp = MPSP(inc=1024, outc=512)
        self.decoder6 = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            BasicBlock(512, 512)
        )

        # ----- Upsample -----
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # ----- Decoder -----
        self.decoder5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            BasicBlock(512, 512)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            BasicBlock(256, 256)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicBlock(128, 128)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64)
        )

        # ----- Side outputs -----
        self.out7 = nn.Conv2d(1024, num_class, 3, padding=1)
        self.out6 = nn.Conv2d(512, num_class, 3, padding=1)
        self.out5 = nn.Conv2d(512, num_class, 3, padding=1)
        self.out4 = nn.Conv2d(256, num_class, 3, padding=1)
        self.out3 = nn.Conv2d(128, num_class, 3, padding=1)
        self.out2 = nn.Conv2d(64, num_class, 3, padding=1)
        self.out1 = nn.Conv2d(64, num_class, 3, padding=1)

        # ----- Refine -----
        self.refunet = RefUnet(num_class, 64)

    def forward(self, x):
        x_in = self.conv_in(x)
        x1 = self.encoder1(x_in)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x6 = self.encoder6(x5)
        psp = torch.cat((self.psp(x6), x6), dim=1)
        y6 = self.decoder6(psp)

        y5 = torch.cat([y6, x5], dim=1)
        y5 = self.decoder5(self.upscore2(y5))
        y4 = torch.cat([y5, x4], dim=1)
        y4 = self.decoder4(self.upscore2(y4))
        y3 = torch.cat([y4, x3], dim=1)
        y3 = self.decoder3(self.upscore2(y3))
        y2 = torch.cat([y3, x2], dim=1)
        y2 = self.decoder2(self.upscore2(y2))
        y1 = torch.cat([y2, x1, x_in], dim=1)
        y1 = self.decoder1(y1)

        out7 = self.upscore16(self.out7(x6))
        out6 = self.upscore16(self.out6(y6))
        out5 = self.upscore8(self.out5(y5))
        out4 = self.upscore4(self.out4(y4))
        out3 = self.upscore2(self.out3(y3))
        out2 = self.out2(y2)
        out1 = self.out1(y1)
        out = self.refunet(out1)

        results = [out7, out6, out5, out4, out3, out2, out1, out]
        results = [F.sigmoid(r) for r in results]
        return results
