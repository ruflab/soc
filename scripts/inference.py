import os
import pprint
import torch
import pytorch_lightning as pl
from soc.val import compute_accs
from soc.losses import compute_losses
from soc.runners import make_runner

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')

ckpt_path = os.path.join(cfd, 'last.ckpt')
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
ckpt['hyper_parameters']['dataset']['dataset_path'] = _DATASET_PATH

pl.seed_everything(ckpt['hyper_parameters']['seed'])

runner = make_runner(ckpt['hyper_parameters'])
runner.setup('fit')
runner.load_state_dict(ckpt['state_dict'])
runner.eval()

pprint.pprint(runner.hparams)

train_dataloader = iter(runner.train_dataloader())
spatial_metadata, linear_metadata, actions_metadata = runner.metadata

print("\n---------------------- batch 1 -------------------\n")
batch1 = next(train_dataloader)
x_seq1 = batch1[0]
y_spatial_s_true_seq1, y_s_true_seq1, y_a_true_seq1 = batch1[1]
y_spatial_s_logits_seq1, y_s_logits_seq1, y_a_logits_seq1 = runner.model(x_seq1)

train_dict1 = {}
train_dict1.update(compute_losses(spatial_metadata, y_spatial_s_logits_seq1, y_spatial_s_true_seq1))
train_dict1.update(compute_losses(linear_metadata, y_s_logits_seq1, y_s_true_seq1))
train_dict1.update(compute_losses(actions_metadata, y_a_logits_seq1, y_a_true_seq1))
pprint.pprint(train_dict1)

val_dict1 = {}
val_dict1.update(compute_accs(spatial_metadata, y_spatial_s_logits_seq1, y_spatial_s_true_seq1))
val_dict1.update(compute_accs(linear_metadata, y_s_logits_seq1, y_s_true_seq1))
val_dict1.update(compute_accs(actions_metadata, y_a_logits_seq1, y_a_true_seq1))
pprint.pprint(val_dict1)

print("\n---------------------- batch 2 -------------------\n")
batch2 = next(train_dataloader)
x_seq2 = batch2[0]
y_spatial_s_true_seq2, y_s_true_seq2, y_a_true_seq2 = batch2[1]
y_spatial_s_logits_seq2, y_s_logits_seq2, y_a_logits_seq2 = runner.model(x_seq2)

train_dict2 = {}
train_dict2.update(compute_losses(spatial_metadata, y_spatial_s_logits_seq2, y_spatial_s_true_seq2))
train_dict2.update(compute_losses(linear_metadata, y_s_logits_seq2, y_s_true_seq2))
train_dict2.update(compute_losses(actions_metadata, y_a_logits_seq2, y_a_true_seq2))
pprint.pprint(train_dict2)

val_dict2 = {}
val_dict2.update(compute_accs(spatial_metadata, y_spatial_s_logits_seq2, y_spatial_s_true_seq2))
val_dict2.update(compute_accs(linear_metadata, y_s_logits_seq2, y_s_true_seq2))
val_dict2.update(compute_accs(actions_metadata, y_a_logits_seq2, y_a_true_seq2))
pprint.pprint(val_dict2)

print("\n---------------------- batch 3 -------------------\n")
batch3 = next(train_dataloader)
x_seq3 = batch3[0]
y_spatial_s_true_seq3, y_s_true_seq3, y_a_true_seq3 = batch3[1]
y_spatial_s_logits_seq3, y_s_logits_seq3, y_a_logits_seq3 = runner.model(x_seq3)

train_dict3 = {}
train_dict3.update(compute_losses(spatial_metadata, y_spatial_s_logits_seq3, y_spatial_s_true_seq3))
train_dict3.update(compute_losses(linear_metadata, y_s_logits_seq3, y_s_true_seq3))
train_dict3.update(compute_losses(actions_metadata, y_a_logits_seq3, y_a_true_seq3))
pprint.pprint(train_dict3)

val_dict3 = {}
val_dict3.update(compute_accs(spatial_metadata, y_spatial_s_logits_seq3, y_spatial_s_true_seq3))
val_dict3.update(compute_accs(linear_metadata, y_s_logits_seq3, y_s_true_seq3))
val_dict3.update(compute_accs(actions_metadata, y_a_logits_seq3, y_a_true_seq3))
pprint.pprint(val_dict3)

breakpoint()
