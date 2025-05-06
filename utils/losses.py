from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all'):
    # def __init__(self, temperature=0.07, contrast_mode='all',
    #              base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature

    # def forward(self, features, labels=None, mask=None):  #256*2*128
    def forward(self, features, proto, labels):  # 256*2*128

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]          #256

        # # compute logits
        class_number = proto.shape[0]
        ones = features.new_ones(batch_size,1)
        zeros = ones.new_zeros(batch_size, class_number)
        labels = labels.unsqueeze(1)
        mask = zeros.scatter_add_(1, labels.long(), ones)



        anchor_dot_contrast = torch.div(
            torch.matmul(features, proto.T),       #512*128 * 128*512自己与自己相乘，512*512
            self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()          #512*512
        #
        # # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)            #512*512
        # # mask-out self-contrast cases
        # logits_mask = torch.scatter(                                #512*512
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask                                   #512*512

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask                #512*512
        exp_logits = torch.exp(anchor_dot_contrast)
        logits = anchor_dot_contrast * mask
        re_logits = logits.sum(1,keepdim = True)
        log_prob = re_logits - torch.log(exp_logits.sum(1, keepdim=True)-torch.exp(re_logits))      #512*512

        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)      #512*512

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos     #512*512
        # loss = - (self.temperature / self.base_temperature) * log_prob  # 512*512
        loss = - log_prob # 512*512
        loss = loss.mean()       #

        return loss
