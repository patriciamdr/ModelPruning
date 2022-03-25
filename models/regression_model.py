import torch

import config
from models.pruned_model import PrunedModel
from models.vgg import VGGNet


class RegressionModel(PrunedModel):
    def __init__(self, hparams, pretrained_model, remove_ratio=config.remove_ratio):
        super(RegressionModel, self).__init__(hparams, pretrained_model, remove_ratio)

        if isinstance(self.pretrained_model, VGGNet):
            self.hparams['lr'] = 0.01

    def forward(self, x):
        # forward pass of pretrained model to get logits
        self.logits = self.pretrained_model.forward(x.to(torch.device(self.hparams['map_location'])))
        return super().forward(x)

    def compute_loss(self, out, targets):
        # loss
        total_loss = sum(((out - self.logits)**2).mean(1)) / out.shape[0]
        return total_loss

    def configure_optimizers(self):
        params = list(self.parameters())
        optim = torch.optim.Adam(params=params, lr=self.hparams['lr'])
        return {'optimizer': optim}
