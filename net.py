import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Custom Conv layers to replace missing pytorch_utils
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True, activation=F.relu):
        super(ConvLayer, self).__init__()
        if isinstance(kernel_size, int):
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
            self.bn = nn.BatchNorm1d(out_channels) if bn else None
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.activation: x = self.activation(x)
        return x

# 2. LFA Fallback (In case LocalFeatureAggregation is missing)
try:
    from LocalFeatureAggregation import LFA
except ImportError:
    class LFA(nn.Module):
        def __init__(self, in_d, out_d):
            super().__init__()
            self.conv = nn.Conv2d(in_d, out_d, 1)
        def forward(self, x, idx): 
            return F.relu(self.conv(x))

class C3Dnet(nn.Module):
    def __init__(self, in_d, out_d):
        super(C3Dnet, self).__init__()
        self.fc0 = ConvLayer(in_d, 64, 1)
        self.block1 = LFA(64, 128)
        self.block2 = LFA(128, 256)
        self.block3 = LFA(256, 512)
        self.block4 = LFA(512, 1024)
        self.dt = ConvLayer(1024, 1024, (1, 1))
        self.d4 = ConvLayer(1024*2, 512, (1, 1))
        self.d3 = ConvLayer(512*2, 256, (1, 1))
        self.d2 = ConvLayer(256*2, 128, (1, 1))
        self.d1 = ConvLayer(128*2, 64, (1, 1))
        self.d0 = ConvLayer(64, out_d, (1, 1))
        
    def forward(self, end_points): 
        xyz, neigh_idx, pool_idx, unsam_idx = end_points
        out0 = self.fc0(xyz[0].permute(0, 2, 1)).unsqueeze(dim=3) 
        out1 = self.block1(out0, neigh_idx[0]); out1p = self.random_sample(out1, pool_idx[0])
        out2 = self.block2(out1p, neigh_idx[1]); out2p = self.random_sample(out2, pool_idx[1])
        out3 = self.block3(out2p, neigh_idx[2]); out3p = self.random_sample(out3, pool_idx[2])
        out4 = self.block4(out3p, neigh_idx[3]); out4p = self.random_sample(out4, pool_idx[3])
        out = self.dt(out4p)
        out = torch.cat((out, out4p), 1); out = self.d4(out)
        out = self.nearest_interp(out, unsam_idx[3]); out = torch.cat((out, out3p), 1); out = self.d3(out)
        out = self.nearest_interp(out, unsam_idx[2]); out = torch.cat((out, out2p), 1); out = self.d2(out)
        out = self.nearest_interp(out, unsam_idx[1]); out = torch.cat((out, out1p), 1); out = self.d1(out)
        out = self.nearest_interp(out, unsam_idx[0]); out = self.d0(out)
        return out
    
    def random_sample(self, feature, pool_idx):
        B, C, N, _ = feature.shape
        idx = pool_idx.transpose(1, 2).expand(B, C, -1)
        return torch.gather(feature.squeeze(3), 2, idx).unsqueeze(3)
    
    def nearest_interp(self, feature, interp_idx):
        B, C, N, _ = feature.shape
        idx = interp_idx.transpose(1, 2).expand(B, C, -1)
        return torch.gather(feature.squeeze(3), 2, idx).unsqueeze(3)

class Siam3DCDNet(nn.Module):
    def __init__(self, in_d, out_d):
        super(Siam3DCDNet, self).__init__()
        self.net = C3Dnet(in_d, 64)
        self.mlp1 = ConvLayer(64, 32, 1)
        self.mlp2 = nn.Conv1d(32, 2, 1)
        
    def forward(self, end_points0, end_points1, knearest_idx):
        out0 = self.net(end_points0)
        out1 = self.net(end_points1)
        k_01, k_10 = knearest_idx
        fout0 = self.feat_diff(out0, out1, k_01)
        fout1 = self.feat_diff(out1, out0, k_10)
        fout0 = self.mlp1(fout0.squeeze(-1)); fout1 = self.mlp1(fout1.squeeze(-1))
        fout0 = self.mlp2(fout0); fout1 = self.mlp2(fout1)
        return F.log_softmax(fout0.transpose(2, 1), dim=-1), F.log_softmax(fout1.transpose(2, 1), dim=-1)
    
    def feat_diff(self, raw, query, nearest_idx):
        B, C, N, _ = raw.shape
        idx = nearest_idx.transpose(1, 2).expand(B, C, -1)
        near = torch.gather(query.squeeze(3), 2, idx).unsqueeze(3)
        return torch.abs(raw - near)
