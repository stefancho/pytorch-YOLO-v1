#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#

import torch.nn as nn
import torch.nn.functional as F
from box_utils import *


class yoloLoss(nn.Module):

    def __init__(self, l_coord, l_noobj):
        super(yoloLoss,self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size(0)

        gt_boxes_mask = target_tensor[:, :, :, 4] == 1
        no_boxes_mask = target_tensor[:, :, :, 4] == 0

        gt_boxes_mask = gt_boxes_mask.unsqueeze(-1).expand_as(target_tensor)  # (N, S, S, Bx5+20=30)
        no_boxes_mask = no_boxes_mask.unsqueeze(-1).expand_as(target_tensor)  # (N, S, S, Bx5+20=30)

        # no objects loss
        pred_no_obj = pred_tensor[no_boxes_mask].view(-1, 30)           # (S*S - M, 30)
        pr_coord_no_obj = pred_no_obj[:, :10].contiguous().view(-1, 5)  # (B*(S*S - M), 5) (: , x1, y1, x2, y2, p)
        no_obj_loss = F.mse_loss(pr_coord_no_obj[:, -1], torch.zeros(pr_coord_no_obj[:, 4].size(), device=pr_coord_no_obj.device), reduction='sum')

        # M is number of 'responsible' grid cells within the batch
        # targets
        gt_boxes = target_tensor[gt_boxes_mask].view(-1, 30)    # (M, 30)
        gt_coord = gt_boxes[:, :10].contiguous().view(-1, 5)    # (B*M, 5) (2*M, x1, y1, x2, y2, p)
        gt_classes = gt_boxes[:, 10:]                           # (M, 20)

        # predictions
        pr_boxes = pred_tensor[gt_boxes_mask].view(-1, 30)      # (M, 30)
        pr_coord = pr_boxes[:, :10].contiguous().view(-1, 5)    # (B*M, 5) (2*M, x1, y1, x2, y2, p)
        pr_classes = pr_boxes[:, 10:]                           # (M, 20)

        # classification loss
        class_loss = F.mse_loss(gt_classes, pr_classes, reduction='sum')


        decoded_targets = decode_coord(gt_coord[:, :4])     # (B*M, 4)
        decoded_preds = decode_coord(pr_coord[:, :4])       # (B*M, 4)

        ious = compute_corresponding_iou(decoded_targets, decoded_preds)  # (B*M,)

        max_ious, max_indx = ious.view(-1, 2).max(dim=1)
        resp_boxes_mask = torch.cat((~max_indx.byte(), max_indx.byte())).view(2, -1).transpose(0, 1).contiguous().view(-1)
        box_pred_resp = pr_coord[resp_boxes_mask, :]
        box_target_resp = gt_coord[resp_boxes_mask, :]
        # localization loss
        loc_loss = F.mse_loss(box_pred_resp[:, :2], box_target_resp[:, :2], reduction='sum')
        loc_loss += F.mse_loss(torch.sqrt(box_pred_resp[:, 2:4]), torch.sqrt(box_target_resp[:, 2:4]), reduction='sum')

        # not 'responsible' predictors confidence loss ???
        # box_pred_not_response = pr_coord[~resp_boxes_mask, :]
        # box_target_not_response = gt_coord[~resp_boxes_mask, :]
        # box_target_not_response[:, 4] = 0
        #
        # not_contain_conf_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # confidence loss
        conf_loss = F.mse_loss(pr_coord[resp_boxes_mask, 4], max_ious, reduction='sum')  # (M,) (M,)

        return (self.l_coord*loc_loss + conf_loss +  self.l_noobj*no_obj_loss + class_loss)/N




