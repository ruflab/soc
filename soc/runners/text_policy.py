import torch
# from torch import Tensor
from . import SOCRunner
from ..val import compute_accs, get_stats
from ..losses import compute_losses


class SOCTextForwardPolicyRunner(SOCRunner):
    """
        A runner represent a training pipeline.
        It contains everything from the dataset to the optimizer.

        Args:
            - config: Hyper parameters configuration
    """
    def training_step(self, batch, batch_nb):
        """
            This function apply an batch update to the model.

            Args:
                - batch: (x, y) batch of data
                - model: (Module) the model
                - metadata: (Dict) metadata to compute losses
        """
        x_seq, x_text_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq, _ = batch[1]

        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = self.model(x_seq, x_text_seq)

        spatial_metadata, linear_metadata, actions_metadata = self.metadata

        train_dict = {}
        train_dict.update(
            compute_losses(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq)
        )
        train_dict.update(compute_losses(linear_metadata, y_s_logits_seq, y_s_true_seq))
        train_dict.update(compute_losses(actions_metadata, y_a_logits_seq, y_a_true_seq))

        loss = torch.tensor(0., device=y_spatial_s_logits_seq.device)
        for k, l in train_dict.items():
            loss += l

        train_dict['train_loss'] = loss

        final_dict = {'loss': train_dict['train_loss'], 'log': train_dict}

        return final_dict

    def validation_step(self, batch, batch_idx):
        """
            This function computes the validation loss and accuracy of the model.

            Args:
                - batch: (x, y) batch of data
                - model: (Module) the model
                - metadata: (Dict) metadata to compute losses
        """

        x_seq, x_text_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq, _ = batch[1]

        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = self.model(x_seq, x_text_seq)

        spatial_metadata, linear_metadata, actions_metadata = self.metadata

        val_dict = {}
        val_dict.update(
            compute_accs(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq)
        )
        val_dict.update(compute_accs(linear_metadata, y_s_logits_seq, y_s_true_seq))
        val_dict.update(compute_accs(actions_metadata, y_a_logits_seq, y_a_true_seq))

        val_acc = torch.tensor(0., device=y_spatial_s_logits_seq.device)
        for k, acc in val_dict.items():
            val_acc += acc
        val_acc = val_acc / len(val_dict)
        val_dict['val_accuracy'] = val_acc

        one_meta = {'piecesonboard_one_mean': spatial_metadata['piecesonboard']}
        val_dict.update(get_stats(one_meta, torch.round(torch.sigmoid(y_spatial_s_logits_seq)), 1))

        return val_dict
