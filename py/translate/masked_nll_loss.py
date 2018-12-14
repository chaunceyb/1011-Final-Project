import torch
import torch.nn as nn
def masked_nll_loss(log_probs, target, length, y_mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    losses = -torch.gather(log_probs, dim=2, index=target.unsqueeze(2)).squeeze(2)
    losses.masked_fill_(1-y_mask, 0) # B x tgt_len
    loss = losses.sum(dim = 1) / length.float()
    return loss