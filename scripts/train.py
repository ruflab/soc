import os
import torch
import torchvision.models as models
from tqdm import tqdm
from soc import SocPSQLDataset
from soc.training import train_on_dataset
from soc.utils import pad_collate_fn

cfd = os.path.dirname(os.path.realpath(__file__))
results_d = os.path.join(cfd, 'results')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = SocPSQLDataset()

model_name = 'resnet18'
pretrained = False
model = models.resnet18(pretrained=pretrained)
model.to(device)

loss_f = torch.nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
n_epochs = 1
batch_size = 4


def end_batch_callback(i_epoch: int, i_batch: int, n_batchs: int, loss: float):
    if i_batch % (n_batchs // 4) == 0:
        tqdm.write(
            "Epoch: {}, {}%, loss: {}".format(i_epoch, round(i_batch / n_batchs * 100), loss)
        )


callbacks = {'end_batch_callback': end_batch_callback}

final_model = train_on_dataset(
    dataset,
    model,
    loss_f,
    optimizer,
    n_epochs,
    batch_size,
    pad_collate_fn,
    callbacks
)

model_save_path = os.path.join(results_d, 'resnet18.pt')
hp_save_path = os.path.join(results_d, 'hp.pt')
hp = {
    'training': {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'optimizer_name': 'Adam',
        'loss_f': 'MSELoss',
        'lr': lr,
    },
    'model': {
        'name': model_name,
        'pretrained': pretrained
    }
}
torch.save(hp, hp_save_path)
torch.save(final_model.state_dict(), model_save_path)
