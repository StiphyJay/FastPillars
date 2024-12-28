import torch.nn as nn
from typing import List
try:
    import spconv.pytorch as spconv
except:
    import spconv
from .pillar_utils import PillarQueryAndGroup, bev_spatial_shape
from .scatter_utils import scatter_max
import torch
import time
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_sum

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import cv2 

def show_feature_map(name,feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().detach().numpy()
    feature_map_num = feature_map.shape[0] #64
    row_num = np.ceil(np.sqrt(feature_map_num)) #64开方
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(int(row_num), int(row_num), index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        cv2.imwrite(str(name)+".png", feature_map[index-1])

class PillarMaxPooling(nn.Module):
    def __init__(self, mlps: List[int], bev_size:float, point_cloud_range:List[float]):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)

        self.groups = PillarQueryAndGroup(bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                # nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                nn.Linear(mlps[k], mlps[k + 1], bias=False),
                nn.BatchNorm1d(mlps[k + 1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, pt_feature):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]
        pillar_indices, pillar_set_indices, group_features, pillar_centers = self.groups(xyz, xyz_batch_cnt, pt_feature) #M*3 L  L*8
        #pillar_indices pillar中有点的pillar坐标，(B,X,Y)
        # group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)
        group_features = self.shared_mlps(group_features)  # (1, C, L) L*8->L*32 group_features特征维度从8->32
        group_features = group_features.transpose(1, 0).contiguous() # 32*L

        pillar_features = scatter_max(group_features, pillar_set_indices, pillar_indices.shape[0])   # (C, M) 32*M  M(83690)个非空pillar，每个pillar的特征维度是32
        pillar_features = pillar_features.transpose(1, 0)   # (M, C)
        #初始化Sparse Tensor  features=pillar_features indices=pillar_indices spatial_shape=(self.bev_height, self.bev_width)  batch_size=B
        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)  #(M, C), (M, 3), (1440,1440), 1

class PillarMaxPooling_dense(nn.Module):
    def __init__(self, mlps: List[int], bev_size:float, point_cloud_range:List[float], use_TA=False, atten_pool=False, use_max=False, leakyrelu=False):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)
        self.use_TA = use_TA
        self.use_attensive_pooling = atten_pool
        self.groups = PillarQueryAndGroup(bev_size, point_cloud_range)
        self.mlps = mlps
        self.use_leakyrelu = leakyrelu
        self.use_max = use_max
        if self.use_leakyrelu:
            act=nn.LeakyReLU()
        else:
            act=nn.ReLU()
        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                # nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                nn.Linear(mlps[k], mlps[k + 1], bias=False),
                nn.BatchNorm1d(mlps[k + 1], eps=1e-3, momentum=0.01),
                act
            ])
        if self.use_attensive_pooling:
            self.score_fn = nn.Sequential(nn.Linear(mlps[-1], mlps[-1]), act)
        self.shared_mlps = nn.Sequential(*shared_mlp)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, pt_feature):
        """
        Args:
            xyz: (N1+N2..., 3) 
            xyz_batch_cnt:  (N1, N2, ...) 
            point_features: (N1+N2..., C) 
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        # torch.cuda.synchronize()
        # start = time.time()
        B = xyz_batch_cnt.shape[0] 
        pillar_indices, pillar_set_indices, group_features, pillar_centers = self.groups(xyz, xyz_batch_cnt, pt_feature) #M*3 L  L*8
        #MAPE module
        group_features = self.shared_mlps(group_features)  # (1, C, L) L*8->L*32
        score_fn = self.score_fn(group_features) #L*C
        attn_score = scatter_softmax(score_fn, pillar_set_indices.to(torch.long), dim=0)
        pillar_features1 = scatter_sum(group_features*attn_score, pillar_set_indices.to(torch.int64), dim=0)
        pillar_features2 = scatter_max(group_features.transpose(1, 0).contiguous(), pillar_set_indices, pillar_indices.shape[0]).transpose(1, 0)
        pillar_features = (pillar_features1 + pillar_features2) / 2
            
        batch_canvas = []
        for batch_itt in range(B):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.mlps[-1],
                self.bev_width * self.bev_height,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            )

            # Only include non-empty pillars
            batch_mask = pillar_indices[:, 0] == batch_itt

            this_coords = pillar_indices[batch_mask, :]
            indices = this_coords[:, 1] * self.bev_height + this_coords[:, 2]
            indices = indices.type(torch.long)
            voxels = pillar_features[batch_mask, :]
            voxels = voxels.transpose(1, 0) 
            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels
            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(B, self.mlps[-1], self.bev_width, self.bev_height)
        return batch_canvas