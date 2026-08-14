from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .utils_motifs import obj_edge_vectors, to_onehot, encode_box_info, nms_overlaps

class GPSNetContext(nn.Module):
    def __init__(self, cfg, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = cfg
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels

        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.GPSNET.HIDDEN_DIM
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.GPSNET.UPDATE_STEP
        self.dropout_rate = cfg.MODEL.ROI_RELATION_HEAD.GPSNET.DROPOUT

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)

        self.bbox_embed = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )

        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.pairwise_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, roi_features, proposals, logger=None):
        rel_pair_inds = cat([proposal.get_field("rel_pair_idxs") for proposal in proposals], dim=0)
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        bbox_info = encode_box_info(proposals)

        pos_embed = self.bbox_embed(bbox_info)
        label_embed = self.obj_embed1(obj_labels)

        obj_pre_feat = cat((roi_features, label_embed, pos_embed), dim=1)
        inst_features = self.lin_obj(obj_pre_feat)
        inst_features = self.dropout(inst_features)

        sub_feat = inst_features[rel_pair_inds[:, 0]]
        obj_feat = inst_features[rel_pair_inds[:, 1]]
        rel_feat = self.pairwise_net(torch.cat([sub_feat, obj_feat], dim=1))

        obj_dists = torch.zeros((inst_features.shape[0], self.num_rel_cls), device=inst_features.device)
        obj_preds = torch.zeros((inst_features.shape[0],), dtype=torch.long, device=inst_features.device)
        edge_ctx = rel_feat

        return obj_dists, obj_preds, edge_ctx
