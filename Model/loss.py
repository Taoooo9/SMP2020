import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothing(nn.Module):

    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.config = config
        self.LogSoftmax = nn.LogSoftmax()

        if self.config.label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False)
        self.confidence = 1.0 - self.config.label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.config.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, logits, labels):
        scores = self.LogSoftmax(logits)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        if self.confidence < 1:
            tdata = labels.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if self.config.use_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(labels.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            labels = tmp_.detach()
        loss = self.criterion(scores, labels)
        return loss


def class_loss(logit, gold):
    batch_size = logit.size(0)
    loss = F.cross_entropy(logit, gold)
    predict_id = torch.max(logit, 1)[1].view(gold.size()).data
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return loss, correct, predict_id, accuracy


def distinguish_loss(logit, gold):
    batch_size = logit.size(0)
    loss = F.cross_entropy(logit, gold)
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return loss, correct, accuracy


def smp_eval(logit):
    predict_id = torch.max(logit, 1)[1]
    return predict_id






