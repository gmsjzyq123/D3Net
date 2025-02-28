import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.instance_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.instance_norm(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.instance_norm(out)
        out += residual
        return out


class SLNet(nn.Module):
    def __init__(self, n_residual_blocks, in_channels, out_channels):
        super(SLNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual_blocks = self.make_residual_blocks(n_residual_blocks, out_channels)
        self.identity_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.fc = nn.Linear(out_channels, out_channels)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.instance_norm = nn.InstanceNorm2d(out_channels)

    def make_residual_blocks(self, n_blocks, channels):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels, channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        # print(x.shape)#[64,64,13,13]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.instance_norm(out)
        out = self.residual_blocks(out)
        out_identity = self.identity_layer(out)
        out_global_avg_pool = self.global_avg_pool(out)
        out_global_avg_pool = out_global_avg_pool.view(out_global_avg_pool.size(0), -1)
        out_global_avg_pool = self.fc(out_global_avg_pool)
        out_global_avg_pool = self.relu(out_global_avg_pool)
        out_global_avg_pool = self.expand(out_global_avg_pool.unsqueeze(2).unsqueeze(3))
        out = out + out_identity
        out = out + out_global_avg_pool
        out = self.tanh(out)
        return out


# class AdaIN2d(nn.Module):
#     def __init__(self, style_dim, num_features):
#         super().__init__()
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         self.fc = nn.Linear(style_dim, num_features * 2)
#
#     def forward(self, x, s):
#         print(x.shape)
#         print(s.shape)
#         h = self.fc(s)
#         print(h.shape)
#         h = h.view(h.size(0), h.size(1), 1, 1)
#         print(h.shape)
#         print(self.norm(x))
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         return (1 + gamma) * self.norm(x) + beta

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # print(x.shape)
        # print(s.shape)
        h = self.fc(s)
        # print(h.shape)
        h = h.view(h.size(0), h.size(1), 1, 1)
        # print(h.shape)
        # print(self.norm(x))
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * x + beta




class Generator(nn.Module):
    def __init__(self, d_se=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10,n_blocks=1, device=0):
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device

        self.adain_morph = AdaIN2d(zdim, d_se)

        self.conv_spa1 = nn.Conv2d(imdim, d_se, 1, 1)
        self.conv_spa2 = nn.Conv2d(64, d_se, 1, 1)

        self.conv_spe1 = nn.Conv2d(imdim, d_se, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(d_se, d_se, imsize[0])

        self.conv1 = nn.Conv2d(d_se + d_se, d_se, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(d_se, imdim, kernelsize, 1, stride)

        self.SLN = SLNet(n_blocks, d_se, d_se)

    def forward(self, x):
        z = torch.randn(len(x), self.zdim).to(self.device)

        x_spa = F.relu(self.conv_spa1(x)) #[64,64,13,13]
        x_spa = self.SLN(x_spa) #[64,64,13,13]
        x_spa = self.conv_spa2(x_spa) #[64,64,13,13]

        x_spe = F.relu(self.conv_spe1(x)) #[64,64,1,1]
        x_spe = self.adain_morph(x_spe, z) #[64,64,1,1]
        x_spe = self.conv_spe2(x_spe) #[64,64,13,13]

        x = F.relu(self.conv1(torch.cat((x_spa, x_spe), 1)))
        x = torch.sigmoid(self.conv2(x))

        return x


# if __name__=='__main__':
#     x = torch.randn(64, 3, 13, 13).to('cuda')
#     G_net = Generator(zdim=10).to('cuda')
#     G_net.eval()
#     y1, y2 = G_net(x)
#     print(y1.shape, y2.shape)
