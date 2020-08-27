import os
import pprint
import torch
import pytorch_lightning as pl
import seaborn as sns
# import matplotlib.pyplot as plt
from soc.runners import make_runner
from soc.datasets import soc_data
# from soc.val import compute_accs
# from soc.losses import compute_losses

sns.set(color_codes=True)

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
# _DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_150_fullseq.pt')
_VAL_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_10_fullseq.pt')

# ckpt_path = os.path.join(cfd, 'last.ckpt')
ckpt_path = os.path.join(cfd, 'soc300_resnet18_pol_1step.ckpt')
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

ckpt['hyper_parameters']['dataset']['dataset_path'] = _DATASET_PATH
ckpt['hyper_parameters']['val_dataset']['dataset_path'] = _VAL_DATASET_PATH
ckpt['hyper_parameters']['dataset']['shuffle'] = False
ckpt['hyper_parameters']['batch_size'] = 1

pl.seed_everything(ckpt['hyper_parameters']['seed'])

runner = make_runner(ckpt['hyper_parameters'])
runner.setup('fit')
runner.load_state_dict(ckpt['state_dict'])
runner.eval()

pprint.pprint(runner.hparams)

spatial_metadata, linear_metadata, actions_metadata = runner.metadata
last_spatial_id = spatial_metadata['piecesonboard'][1]
dice_distrib_idx = linear_metadata['diceresult']
with torch.no_grad():
    for batch in runner.train_dataloader():
        x_seq = batch[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]
        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = runner.model(x_seq)

        last_action_idx = int(torch.argmax(x_seq[0][-1][-soc_data.ACTION_SIZE:, 0, 0]).item())
        last_action_name = soc_data.ACTIONS_NAMES[last_action_idx]
        previous_dice_idx = int(
            torch.argmax(
                x_seq[0][-1][last_spatial_id + dice_distrib_idx[0]:last_spatial_id
                             + dice_distrib_idx[1],
                             0,
                             0].item()
            )
        )
        previous_dice_val = soc_data.DICE_RESULTS[previous_dice_idx]
        future_dice_idx = int(
            torch.argmax(y_s_true_seq[0][0][dice_distrib_idx[0]:dice_distrib_idx[1]]).item()
        )
        future_dice_val = soc_data.DICE_RESULTS[future_dice_idx]

        dice_logits_preds = y_s_logits_seq[0][0][dice_distrib_idx[0]:dice_distrib_idx[1]]
        dice_distrib_preds = torch.softmax(dice_logits_preds, dim=0)
        dice_output_preds = soc_data.DICE_RESULTS[int(torch.argmax(dice_distrib_preds).item())]

        print(
            "{} {} -> {} || predicted: {}".format(
                last_action_name, previous_dice_val, future_dice_val, dice_output_preds
            )
        )
        if last_action_name == 'ROLL':
            dice_values = [-1, 0] + list(range(2, 13))
            sns.barplot(x=dice_values, y=dice_distrib_preds)
            # breakpoint()
            # plt.show()
