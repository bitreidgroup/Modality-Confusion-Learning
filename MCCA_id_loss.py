from __future__ import absolute_import

import torch
from torch import nn


class MCCALossID(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=395, feat_dim=2048, rho = 4500,sigma = 8, use_gpu=True):
        super(MCCALossID, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.rho = rho
        self.sigma = sigma
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = x.size(0)
        labels=labels.long()
        print(type(labels[1]))
        center_batch = self.centers[labels, :]
        center_square = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        centermat = torch.pow(center_batch, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + center_square
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + center_square
        centermat.addmm_(center_batch, self.centers.t(), alpha=-2)
        distmat.addmm_(x, self.centers.t(), alpha=-2) # distmat: (batch_size, num_classes)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        val, _ = torch.min(centermat[~mask].view(batch_size, self.num_classes-1), dim=1) 
        dist = distmat * mask.float()
        loss_push = max(self.rho - (val.clamp(min=1e-12, max=1e+12).sum()) / batch_size,0)
        #nearnest embedding nearest
        dist1 = dist.permute(1, 0)
        nearest = 0
        for i in dist1:
            if i[torch.nonzero(i)].size(0):
                nearest += torch.min(i[torch.nonzero(i).detach()])
        loss_nearest = nearest * 8 / batch_size

        loss_center = max(dist.clamp(min=1e-12, max=1e+12).sum() / batch_size, 0)

        loss_1 = max(loss_center - loss_nearest - self.sigma, 0)
        return  loss_1 + loss_push


if __name__ == '__main__':
    import torch.optim as optim
    use_gpu = True
    mcca_id_loss = MCCALossID(use_gpu=use_gpu)
    features = torch.rand(64, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(64, 2048).cuda()
        targets = torch.Tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
             4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]).long().cuda()
    optimizer = optim.SGD(mcca_id_loss.parameters(), lr=0.001, momentum=0.9)

    for i in range(10):
        optimizer.zero_grad()
        loss = mcca_id_loss(features, targets)
        loss.backward()
        optimizer.step()
        print(loss)
