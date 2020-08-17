import os
import soc
from dataclasses import dataclass
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from soc.datasets import PSQLConfig

cfd = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cfd, '..', 'data')
fixture_dir = os.path.join(cfd, '..', 'tests', 'fixtures')


@dataclass
class DumpConfig(PSQLConfig):
    testing: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=DumpConfig)

if __name__ == "__main__":
    with initialize():
        config = compose(config_name="config")
        if config.testing is True:
            ds = soc.datasets.SocPSQLSeqDataset(config)
            ds.dump_preprocessed_dataset(fixture_dir, True)
        else:
            ds = soc.datasets.SocPSQLSeqDataset(config)
            ds.dump_preprocessed_dataset(data_folder)
