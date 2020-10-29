import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d = d / (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)
    return d


class VATLoss(nn.Module):

    def __init__(self, config):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.config = config
        self.xi = config.vat_xi
        self.eps = config.vat_eps
        self.ip = config.vat_iter

    def forward(self, model, x):
        with torch.no_grad():
            ul_logit = model(x)
            pred = F.softmax(ul_logit[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(ul_logit[1].shape).sub(0.5)
        if self.config.use_cuda:
            d = d.cuda()
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x, vat=self.xi * d)
                pred_hat = pred_hat[0]
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x, vat=r_adv)
            pred_hat = pred_hat[0]
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
