import math
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init

from .bbox_utils import decode, nms, match, log_sum_exp, match_ssd


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]

        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE or [0.1]
        # self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP 
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps

    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class Detect(object):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf scores
    and threshold to a top_k number of output predictions for both confidence
    score and locations.
    """
    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (tensor)
                Shape: [batch * num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1, num_priors, 4]
        """

        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)

        return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class MultiBoxLoss(nn.Module):
	"""SSD Weighted Loss Function

	Compute Targets:
            1) Produce Confidence Target Indices by matching ground truth boxes
                with (default) 'priorboxes' that have jaccard index > threshold parameter
                (default threshold: 0.5).
            2) Produce localization target by 'encoding' variance into offsets of ground
                truth boxes and their matched 'priorboxes'.
            3) Hard negative mining to filter the excessive number of negative examples
                that comes with using a large number of default bounding boxes.
                (default negative: positive ratio 3:1)
	
	Objective Loss:
            L(x, c, l, g) = (Lconf(x, c) + alphaLloc(x, l, g)) / N
            Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
            weighted by alpha which is set to 1 by cross val.
            Args:
                c: class confidences,
                l: predicted boxes,
                g: ground truth boxes,
                N: number of matched default boxes
            See: https://arxiv.org/pdf/1512.02325.pdf for more details.
		
	"""
	def __init__(self, cfg, dataset, use_gpu=True):
            super(MultiBoxLoss, self).__init__()
            self.use_gpu = use_gpu
            self.num_classes = cfg.NUM_CLASSES
            self.negpos_ratio = cfg.NEG_POS_RATIOS
            self.variance = cfg.VARIANCE
            self.dataset = dataset

            if dataset == 'face':
                self.threshold = cfg.FACE.OVERLAP_THRESH
                self.match = match
            elif dataset == 'hand':
                self.threshold = cfg.HAND.OVERLAP_THRESH
                self.match = match_ssd
            else:
                self.threshold = cfg.HEAD.OVERLAP_THRESH
                self.match = match

	def forward(self, predictions, targets):
            """Multibox Loss
            
            Args:
                    predictions (tuple): A tuple containing loc preds, conf preds, and prior boxes from SDD net.
                            loc shape : torch.size(batch_size, num_priors, 4)	
                            conf shape: torch.size(batch_size, num_priors, num_classes)
                            priors shape: torch.size(num_priors, 4)

                    targets (tensor): Ground Truth boxes and labels for a batch.
                            shape: torch.size(batch_size, num_objs, 5) (last idx is the label).

            """

            loc_data, conf_data, priors = predictions
            num = loc_data.size(0)
            priors = priors[:loc_data.size(1), :]
            num_priors = (priors.size(0))
            num_classes = self.num_classes

            # match priors(default boxes) and ground truth boxes
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.LongTensor(num, num_priors)
            for idx in range(num):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                defaults = priors.data
                self.match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
