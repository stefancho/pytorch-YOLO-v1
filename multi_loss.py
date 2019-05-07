#encoding:utf-8
#
#created by stefancho 4.05.2019
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from box_utils import *


class MultiLoss(nn.Module):

    def __init__(self, l_coord, l_noobj):
        super(MultiLoss,self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+40=50) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,50)
        '''
        N = pred_tensor.size(0)
        device = pred_tensor.device

        left_boxes_mask = target_tensor[:, :, :, 4] == 1
        right_boxes_mask = target_tensor[:, :, :, 29] == 1

        # all grid cells without objects
        no_boxes_mask = ~(left_boxes_mask | right_boxes_mask)
        no_boxes_mask = no_boxes_mask.unsqueeze(-1).expand_as(target_tensor)  # (N, S, S, Bx5+40=50)

        # no objects loss
        pred_no_obj = pred_tensor[no_boxes_mask].view(-1, 25)
        pr_coord_no_obj = pred_no_obj[:, :5].contiguous().view(-1, 5)
        no_obj_confidence = torch.zeros(pr_coord_no_obj[:, -1].size(), device=device)
        no_obj_loss = F.mse_loss(pr_coord_no_obj[:, -1], no_obj_confidence, reduction='sum')

        left_boxes_mask = left_boxes_mask.unsqueeze(-1).expand_as(target_tensor).contiguous()
        left_boxes_mask[:, :, :, 25:] = 0
        right_boxes_mask = right_boxes_mask.unsqueeze(-1).expand_as(target_tensor).contiguous()
        right_boxes_mask[:, :, :, :25] = 0

        resp_boxes_mask = left_boxes_mask | right_boxes_mask

        # targets
        target_boxes = target_tensor[resp_boxes_mask].view(-1, 25)
        target_coord = target_boxes[:, :5].contiguous().view(-1, 5)
        target_classes = target_boxes[:, 5:]

        # predictions
        pr_boxes = pred_tensor[resp_boxes_mask].view(-1, 25)
        pr_coord = pr_boxes[:, :5].contiguous().view(-1, 5)
        pr_classes = pr_boxes[:, 5:]

        # localization loss
        loc_loss = F.mse_loss(pr_boxes[:, :2], target_boxes[:, :2], reduction='sum')
        loc_loss += F.mse_loss(torch.sqrt(pr_boxes[:, 2:4]), torch.sqrt(target_boxes[:, 2:4]), reduction='sum')

        decoded_targets = decode_coord(target_coord[:, :4])
        decoded_preds = decode_coord(pr_coord[:, :4])

        ious = compute_corresponding_iou(decoded_targets, decoded_preds)

        # confidence loss
        conf_loss = F.mse_loss(pr_boxes[:, 4], ious, reduction='sum')

        # classification loss
        class_loss = F.mse_loss(target_classes, pr_classes, reduction='sum')

        return (self.l_coord*loc_loss + 2*conf_loss + self.l_noobj*no_obj_loss + class_loss)/N

