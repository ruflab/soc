from . import datasets
from . import models
from hydra.core.config_store import ConfigStore

__all__ = ['datasets', 'models']

cs = ConfigStore.instance()
cs.store(group="generic/model", name="convlstm", node=models.ConvLSTMConfig)
cs.store(group="generic/model", name="conv3d", node=models.Conv3dModelConfig)
cs.store(group="generic/model", name="resnet18", node=models.ResNetConfig)
cs.store(group="generic/dataset", name="psqlseqsatos", node=datasets.PSQLConfig)
cs.store(
    group="generic/dataset",
    name="preprocessedforwardsatosa",
    node=datasets.PreprocessedForwardConfig
)
cs.store(
    group="generic/dataset",
    name="preprocessedseqsatosapolicy",
    node=datasets.PreprocessedSeqConfig
)
