import torch
import torch.nn as nn


__configs = {
    'name': 'hoc',
    'fixed': True,
    'create_old_model': True,
    'preallocated_classes': 1024,
    'mu_': 10,
    'lambda_': 0.1
}


def HOCconfigs():
    return __configs


class HocLoss(nn.Module):
    def __init__(self, mu_):
        super(HocLoss, self).__init__()
        self.mu_ = mu_
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
               ):
        loss = self._loss(feat_old, feat_new, labels)
        return loss

    def _loss(self, out0, out1, labels):
        """Calculates infoNCE loss.
        This code implements Equation 4 in "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements"

        Args:
            feat_old:
                features extracted with the old model.
                Shape: (batch_size, embedding_size)
            feat_new:
                features extracted with the new model.
                Shape: (batch_size, embedding_size)
            labels:
                Labels of the images.
                Shape: (batch_size,)
        Returns:
            Mean loss over the mini-batch.
        """
        ## create diagonal mask that only selects similarities between
        ## representations of the same images
        batch_size = out0.shape[0]
        diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", out0, out1) *  self.mu_

        positive_loss = -sim_01[diag_mask]
        # Get the labels of out0 and out1 samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01* (~class_mask)).view(batch_size, -1)

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()

