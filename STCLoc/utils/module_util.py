import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_util import sample_and_group, sample_and_group_all, similarity

# ===========================================
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"
# ===========================================
import time

import torch
import torch.nn as nn

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )
        self.device = device

        self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        # 输入就是 xyz 所以 input 的维度是 [B, N, 3]
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu() # [B, N, 3] cpu
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1) # [B, N, 3] -> [B, N, 8] 
        x = self.bn_start(x) # [B, 8, N, 1]

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []
        
        # 这段代码使用了 Python 的 torch.randperm 函数来生成一个随机排列的索引
        # 所以这是在打乱点云中的点
        permutation = torch.randperm(N) 
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x) # [B, N, 3], [B, 8, N, 1] -> [B, 2*d_out, N, 1]
            x_stack.append(x.clone())
            decimation_ratio *= d # = 4, 16, 64, 256
            x = x[:,:,:N//decimation_ratio] # [B, 2*d_out, N, 1] -> [B, 2*d_out, N//de...ratio, 1]

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x) # [B, 512, N//256, 1] -> [B, 512, N//256, 1]

        # <<<<<<<<<< DECODER
        # for mlp in self.decoder:
        #     neighbors, _ = knn(
        #         coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
        #         coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
        #         1
        #     ) # shape (B, N, 1)
        #     neighbors = neighbors.to(self.device)

        #     extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

        #     x_neighbors = torch.gather(x, -2, extended_neighbors)

        #     x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

        #     x = mlp(x)

        #     decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        # x = x[:,:,torch.argsort(permutation)]

        # scores = self.fc_end(x)

        # return scores.squeeze(-1)
        return x

# ==================================================================================


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, normalize_radius=False, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        """
        npoint: keyponts number to sample
        radius: sphere radius in a group
        nsample: how many points to group for a sphere
        in_channel: input dimension
        mlp: a list for dimension changes
        normalize_radius: scale normalization
        group_all: wheather use group_all or not
        """
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.normalize_radius = normalize_radius
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        # D' 的值与 out_channel 有关，即 mlp 的最后一个值
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel     

    def forward(self, xyz, points):
        if self.group_all:
            # group_all == True 走这条分支
            new_xyz, new_points = sample_and_group_all(xyz, points) 
            # [B, 1, C], [B, 1, N, C+D]
        else:
            # group_all == False 走这条分支
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.normalize_radius) 
            # [B, npoint, C], [B, npoint, nsample, C+D]
        
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]  or [B, C+D, N, 1]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0] 
        new_points = new_points.permute(0, 2, 1) 

        return new_xyz, new_points


class FeatureCorrelation(nn.Module):
    def __init__(self, steps, feat_size):
        super(FeatureCorrelation, self).__init__()
        self.steps = steps       
        self.feat_size = feat_size
        # ==========================
        # self.feat_size = 8192
        # feat_size = 8192
        # ==========================
        self.pos_embedding = nn.Parameter(torch.randn(1, steps, feat_size))   

    def forward(self, feat_in):
        """
        TAFA
        Input:
            feat_in: tensor([B*T, D])
        Return:
            feat_out: tensor([B*T, D])
        """
        B                   = feat_in.size(0) // self.steps  # B
        feat_in             = feat_in.view(B, self.steps, self.feat_size)  # [B, T, D]
        feat_in             = feat_in + self.pos_embedding   # [B, T, D]
        feat1, feat2, feat3 = torch.split(feat_in, 1, dim=1)  # [B, 1, D]*2
        feat1_new           = similarity(feat1, feat2, feat3)  # [B, 1, D]
        feat2_new           = similarity(feat2, feat1, feat3)  # [B, 1, D]
        feat3_new           = similarity(feat3, feat1, feat2)  # [B, 1, D]         
        feat_out            = torch.cat((feat1_new, feat2_new, feat3_new), dim=1)  # (B, T, D)
        feat_out = feat_out.view(B*self.steps, self.feat_size)   # [B*T, D]

        return feat_out

class PCLocEncoder(nn.Module):
    def __init__(self, num_classes, steps=2, feature_correlation=False, num_neighbors=16, decimation=4, device=torch.device('cuda')):
        super(PCLocEncoder, self).__init__()
        # PointNetSetAbstraction   init 函数签名
        # def __init__(self, npoint, radius, nsample, in_channel, mlp, normalize_radius=False, group_all=False):

        # oxford
        # self.sa1 = PointNetSetAbstraction(512,  4,    32,   3,       [32, 32, 64],     False, False)
        # self.sa2 = PointNetSetAbstraction(128,  8,    16,   64 + 3,  [64, 128, 256],   False, False)

        # # vreloc
        # # self.sa1 = PointNetSetAbstraction(512,  0.2,    32,   3,       [32, 32, 64],     False, False)
        # # self.sa2 = PointNetSetAbstraction(128,  0.4,    16,   64 + 3,  [64, 128, 256],   False, False)

        # self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], False, True)
        self.correlation = FeatureCorrelation(steps, 1024)
        self.feature_correlation = feature_correlation


        # ========================================================
        # 用来替换 sa1, sa2, sa3 这三层
        # 1. sa1, sa2 中 group_all == False， sa3 中 group_all == True 
        # 2. mlp 参数的最后一个值为 1024
        # 所以 xyz -> l3_points | [B, N, 3] -> [B, 512, 64] -> [B, 128, 256] -> [B, 1, 1024]
        # 所以 RandLANet 要替代它们也应该实现将 [B, N, 3] 的输入处理成 [B, 1, 1024] 才对
        # 
        # 我们把 RandLANet 的编码层和输出层去掉了
        # 尝试方案一，用 xyz 填充 rgb 特征
        # 艹，好像 d_in = 3 + d，那就不用什么尝试了，
        # 1. 输入是 [B, N, d_in] => d_in = 3
        # 2. 输出是 [B, 512, N//256, 1]
        # STCLoc 对每帧点云只要提一个全局特征， 
        # 得先过一个 polling 把 N // 256 变成 1
        # 训练时 N == 4096 所以我们得到的结果是 8192 （= 4096//256 * 512)
        # 测试时 N != 4096 所以我们硬编码 8192 得到错误结果
        self.Encoder = RandLANet(3, num_classes, num_neighbors, decimation, device)
        self.linear  = nn.Linear(512, 1024)
        # ========================================================

    def forward(self, xyz):
        B                 = xyz.size(0) # 这里 B 就是 B*T
        # l1_xyz, l1_points = self.sa1(xyz, None)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l3_points         = l3_points.view(B, -1)  # [B*T, D]

        # 为什么上面是 B*T, 这个 T 还要搞明白
        # 搞明白了，输入的就是 [B*T, N, 3], 参见 train.py 
        # 所以原来过完 sa3 得到的结果是 [B*T, 1024]
        # assert xyz.size(1) == 4096, f"目前的维度是 f{xyz.size(1)}."
        l3_points = self.Encoder(xyz) # [B*T, 512, N//256, 1]
        l3_points = torch.max(l3_points, dim=2, keepdim=True)[0].view(B, -1) # [B*T, 512, 1, 1] -> [B*T, 512]
        l3_points = self.linear(l3_points) # [B*T, 512] -> [B*T, 1024]
        print(l3_points.shape)

        if self.feature_correlation:
            l3_points     = self.correlation(l3_points)  # [B*T, D]  TAFA

        return l3_points


class PCLocDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PCLocDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        # ==========================
        # last_channel = 8192
        # ==========================
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  # [B, D]
        
        return x