import collections
import torch
from torch import Tensor
from . import SOCRunner
from ..val import compute_accs, get_stats
from ..losses import compute_losses


class SOCForwardPolicyRunner(SOCRunner):
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
        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = self.model(x_seq)

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

        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = self.model(x_seq)

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


class SOCSeqPolicyRunner(SOCRunner):
    """
        A runner represent a training pipeline.
        It contains everything from the dataset to the optimizer.

        Args:
            - config: Hyper parameters configuration
    """
    def training_step(self, batch, batch_nb, hidden_state=None):
        """
            This function apply an batch update to the model.

            Args:
                - batch: (x, y, mask) batch of data
                - model: (Module) the model
                - metadata: (Dict) metadata to compute losses
        """
        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

        mask_spatial, mask_linear, mask_action = batch[2]

        # We assume the model outputs a tuple where the first element
        # is the actual predictions
        outputs, hidden_state, _ = self.model(x_seq, hidden_state)
        y_spatial_s_logits_seq_raw, y_s_logits_seq_raw, y_a_logits_seq_raw = outputs
        y_spatial_s_logits_seq = y_spatial_s_logits_seq_raw * mask_spatial
        y_s_logits_seq = y_s_logits_seq_raw * mask_linear
        y_a_logits_seq = y_a_logits_seq_raw * mask_action

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

        final_dict = {
            'loss': train_dict['train_loss'],
            'log': train_dict,
            'hiddens': self.detach_hidden_state(hidden_state),
        }

        return final_dict

    def detach_hidden_state(self, hidden_state=None):
        if hidden_state is None:
            return None

        if isinstance(hidden_state, torch.Tensor):
            return hidden_state.detach()

        hidden_state_l = []
        for s in hidden_state:
            if isinstance(s, torch.Tensor):
                hidden_state_l.append(s.detach())
            elif isinstance(s, collections.Sequence):
                sub_l = []
                for sub_s in s:
                    sub_l.append(sub_s.detach())
                hidden_state_l.append(sub_l)

        return hidden_state_l

    def validation_step(self, batch, batch_idx):
        """
            This function apply an batch update to the model.

            Args:
                - batch: (x, y, mask) batch of data
                - model: (Module) the model
                - metadata: (Dict) metadata to compute losses
        """
        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

        mask_spatial, mask_linear, mask_action = batch[2]

        # We assume the model outputs a tuple where the first element
        # is the actual predictions
        outputs = self.model(x_seq)
        y_spatial_s_logits_seq_raw, y_s_logits_seq_raw, y_a_logits_seq_raw = outputs[0]
        y_spatial_s_logits_seq = y_spatial_s_logits_seq_raw * mask_spatial
        y_s_logits_seq = y_s_logits_seq_raw * mask_linear
        y_a_logits_seq = y_a_logits_seq_raw * mask_action

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

    def tbptt_split_batch(self, batch: Tensor, split_size: int) -> list:
        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]
        mask_spatial, mask_linear, mask_action = batch[2]

        time_dims = [
            len(x_seq[0]),
            len(y_spatial_s_true_seq[0]),
            len(y_s_true_seq[0]),
            len(y_a_true_seq[0]),
            len(mask_spatial[0]),
            len(mask_linear[0]),
            len(mask_action[0]),
        ]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"

        n_timesteps = time_dims[0]
        splits = []
        for t in range(0, n_timesteps, split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    split_x = x[:, t:t + split_size]
                elif isinstance(x, collections.Sequence):
                    # Beware!
                    # The following code has been changed from the original function
                    # pytorch_lightning.core.LightningModule.tbptt_split_batch().
                    # We consider a list of batched outputs and not a list of sequences here
                    split_x = [None] * len(x)  # type:ignore
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][:, t:t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits
