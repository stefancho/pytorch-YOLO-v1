import torch
from config import yolo


def decode_coord(encoded_boxes):
    """
        Transfers from grid-wise back to image-wise coordinates
        encoded_boxes: tensor [N, 4] with values [0, 1]
        :returns tensor [N, 4]
    """
    grid_num = yolo['grid_num']
    cell_size = 1. / grid_num

    return torch.cat((encoded_boxes[:, :2] * cell_size - .5 * encoded_boxes[:, 2:4],
                      encoded_boxes[:, :2] * cell_size + .5 * encoded_boxes[:, 2:4]), 1)


def compute_corresponding_iou(box1, box2):
    '''Compute the corresponding intersection over union of two list of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [N,4].
    Return:
      (tensor) iou, sized [N,].
    '''
    N = box1.size(0)
    M = box2.size(0)

    assert N == M

    lt = torch.max(
        box1[:, :2],  # [N,2]
        box2[:, :2],  # [N,2]
    )

    rb = torch.min(
        box1[:, 2:],  # [N,2]
        box2[:, 2:],  # [N,2]
    )

    wh = rb - lt  # [N,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, 0] * wh[:, 1]  # [N]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N,]

    iou = inter / (area1 + area2 - inter)
    return iou


    # def compute_iou(self, box1, box2):
    #     '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    #     Args:
    #       box1: (tensor) bounding boxes, sized [N,4].
    #       box2: (tensor) bounding boxes, sized [M,4].
    #     Return:
    #       (tensor) iou, sized [N,M].
    #     '''
    #     N = box1.size(0)
    #     M = box2.size(0)
    #
    #     lt = torch.max(
    #         box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
    #         box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    #     )
    #
    #     rb = torch.min(
    #         box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
    #         box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    #     )
    #
    #     wh = rb - lt  # [N,M,2]
    #     wh[wh<0] = 0  # clip at 0
    #     inter = wh[:,:,0] * wh[:,:,1]  # [N,M]
    #
    #     area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    #     area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    #     area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    #     area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
    #
    #     iou = inter / (area1 + area2 - inter)
    #     return iou