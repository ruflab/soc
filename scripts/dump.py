import os
import soc
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from soc.datasets import PSQLConfig

cfd = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cfd, '..', 'data')
fixture_dir = os.path.join(cfd, '..', 'tests', 'qfixtures')


@dataclass
class DumpConfig(PSQLConfig):
    raw: bool = False
    testing: bool = False
    separate_seq: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=DumpConfig)


@hydra.main(config_name='config')
def dump(config):
    if config.testing:
        folder = fixture_dir
    else:
        folder = data_folder

    ds = soc.datasets.SocPSQLSeqDataset(config)

    if config.raw:
        ds.dump_raw_dataset(folder)
    else:
        ds.dump_preprocessed_dataset(folder, config.testing, config.separate_seq)


if __name__ == "__main__":
    dump()
