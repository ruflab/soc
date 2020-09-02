import torch
from . import SOCRunner
from ..val import compute_accs, get_stats
from ..losses import compute_losses


class SOCSupervisedSeqRunner(SOCRunner):
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
                - batch: (x, y, mask) batch of data
                - model: (Module) the model
                - metadata: (Dict) metadata to compute losses
        """
        x = batch[0]
        y_true = batch[1]
        mask = batch[2]

        # We assume the model outputs a tuple where the first element
        # is the actual predictions
        outputs = self.model(x)
        y_logits_raw = outputs[0]
        y_logits = y_logits_raw * mask

        train_dict = compute_losses(self.output_metadata, y_logits, y_true)

        loss = torch.tensor(0., device=y_logits.device)
        for k, l in train_dict.items():
            loss += l
        train_dict['train_loss'] = loss

        final_dict = {'loss': train_dict['train_loss'], 'log': train_dict}

        return final_dict

    def validation_step(self, batch, batch_idx):
        """This function computes the validation loss and accuracy of the model."""

        x = batch[0]
        y_true = batch[1]
        mask = batch[2]

        # We assume the model outputs a tuple where the first element
        # is the actual predictions
        outputs = self.model(x)
        y_logits_raw = outputs[0]
        y_logits = y_logits_raw * mask

        val_dict = compute_accs(self.output_metadata, y_logits, y_true)

        val_acc = torch.tensor(0., device=y_logits.device)
        for k, acc in val_dict.items():
            val_acc += acc
        val_acc = val_acc / len(val_dict)
        val_dict['val_accuracy'] = val_acc

        one_meta = {'piecesonboard_one_mean': self.output_metadata['piecesonboard']}
        if 'actions' in self.output_metadata.keys():
            one_meta['actions_one_mean'] = self.output_metadata['actions']
        val_dict.update(get_stats(one_meta, torch.round(y_logits), 1))

        return val_dict


class SOCSupervisedForwardRunner(SOCRunner):
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
        x = batch[0]
        y_true = batch[1]

        y_logits = self.model(x)

        train_dict = compute_losses(self.output_metadata, y_logits, y_true)

        loss = torch.tensor(0., device=y_logits.device)
        for k, l in train_dict.items():
            loss += l
        train_dict['train_loss'] = loss

        final_dict = {'loss': train_dict['train_loss'], 'log': train_dict}

        return final_dict

    def validation_step(self, batch, batch_idx):
        """This function computes the validation loss and accuracy of the model."""

        x = batch[0]
        y_true = batch[1]

        y_logits = self.model(x)

        val_dict = compute_accs(self.output_metadata, y_logits, y_true)

        val_acc = torch.tensor(0., device=y_logits.device)
        for k, acc in val_dict.items():
            val_acc += acc
        val_acc = val_acc / len(val_dict)
        val_dict['val_accuracy'] = val_acc

        out_meta_keys = self.output_metadata.keys()
        if 'mean_piecesonboard' in out_meta_keys:
            prefix = 'mean_'
        else:
            prefix = ''
        one_meta = {'piecesonboard_one_mean': self.output_metadata[prefix + 'piecesonboard']}
        if 'actions' in out_meta_keys or 'mean_actions' in out_meta_keys:
            one_meta['actions_one_mean'] = self.output_metadata[prefix + 'actions']
        val_dict.update(get_stats(one_meta, torch.round(y_logits), 1))

        return val_dict
