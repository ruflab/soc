import os
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from typing import Callable, List, Any, Dict, Optional
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig, OmegaConf
from .runners import make_runner

CollateFnType = Callable[[List[Any]], Any]


@dataclass
class GenericConfig:
    seed: int = 1
    verbose: bool = False
    runner_name: str = MISSING
    dataset: Any = MISSING
    model: Any = MISSING
    lr: float = 3e-3
    optimizer: str = 'adam'
    scheduler: Optional[str] = None
    batch_size: int = 32
    weight_decay: Optional[float] = 0.


@dataclass
class SocConfig:
    defaults: List[Any] = MISSING

    generic: GenericConfig = GenericConfig()
    trainer: Any = MISSING
    other: Any = MISSING


def custom_lr_finder(runner: LightningModule, omegaConf: DictConfig) -> LightningModule:
    """
        LR finder
        The prepare_data/setup does not work well with the lr finder
        To handle the situation we manually search for it before the training

        The situation is being handled:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/2485
    """
    del omegaConf['trainer']['auto_lr_find']
    tmp_trainer = pl.Trainer(**omegaConf['trainer'])
    runner.prepare_data()
    runner.setup('lr_finder')
    lr_finder = tmp_trainer.lr_find(runner)
    # fig = lr_finder.plot(suggest=True)
    new_lr = lr_finder.suggestion()
    omegaConf['generic']['lr'] = new_lr
    runner = make_runner(omegaConf['generic'])

    if omegaConf['generic'].get('verbose', False) is True:
        print('Learning rate found: {}'.format(new_lr))

    return runner


def build_checkpoint_callback(config: Dict) -> ModelCheckpoint:
    save_top_k = 1
    period = 1
    if 'other' in config:
        if 'save_top_k' in config['other']:
            save_top_k = config['other']['save_top_k']
        if 'period' in config['other']:
            period = config['other']['period']
    checkpoint_callback = ModelCheckpoint(
        filepath=config['trainer']['default_root_dir'],
        verbose=True,
        save_top_k=save_top_k,
        save_last=True,
        monitor='val_accuracy',
        mode='max',
        period=period
    )

    return checkpoint_callback


def build_logger(config: Dict):
    if config['trainer']['logger'] == 'neptune':
        logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project_name=os.environ['NEPTUNE_PROJECT_NAME'],
            params=config['generic'],
        )
    else:
        raise ValueError('Logger {} unknown'.format(config['trainer']['logger']))

    return logger


def train(omegaConf: DictConfig) -> LightningModule:
    # Misc part
    if omegaConf['generic']['verbose'] is True:
        print(omegaConf.pretty())

    pl.seed_everything(omegaConf['generic']['seed'])

    # Runner part
    runner = make_runner(omegaConf['generic'])

    if "auto_lr_find" in omegaConf['trainer'] and omegaConf['trainer']['auto_lr_find'] is True:
        runner = custom_lr_finder(runner, omegaConf)

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    if 'default_root_dir' not in config['trainer'] or config['trainer']['default_root_dir'] is None:
        cfd = os.path.dirname(os.path.realpath(__file__))
        default_results_dir = os.path.join(cfd, '..', 'scripts', 'results')
        config['trainer']['default_root_dir'] = default_results_dir

    config['trainer']['checkpoint_callback'] = build_checkpoint_callback(config)

    if 'logger' in config['trainer']:
        config['trainer']['logger'] = build_logger(config)

    if 'deterministic' in config['trainer']:
        config['trainer']['deterministic'] = True

    # ###
    # # Early stopping
    # # It is breaking neptune logging somehow, it seems that it overrides by 1 the current timestep
    # ###
    # early_stop_callback = EarlyStopping(
    #     monitor='val_accuracy', min_delta=0.00, patience=10, verbose=False, mode='max'
    # )
    # config['trainer']['early_stop_callback'] = early_stop_callback

    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(runner)

    return runner
