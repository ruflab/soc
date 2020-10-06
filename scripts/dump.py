import os
import soc
from dataclasses import dataclass
import hydra
from typing import Optional
# from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from soc.datasets import PSQLConfig

cfd = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cfd, '..', 'data')
fixture_dir = os.path.join(cfd, '..', 'tests', 'fixtures')


@dataclass
class DumpConfig(PSQLConfig):
    psql_password: str = 'dummy'

    folder: Optional[str] = None

    raw: bool = False
    testing: bool = False
    separate_seq: bool = False
    dump_text: bool = False

    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = False
    set_empty_text_to_zero: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=DumpConfig)


@hydra.main(config_name='config')
def dump(config):
    if config.folder is None:
        if config.testing:
            config.folder = fixture_dir
        else:
            config.folder = data_folder

    if config.dump_text:
        ds = soc.datasets.SocPSQLTextBertSeqDataset(config)
    else:
        ds = soc.datasets.SocPSQLSeqDataset(config)

    if config.raw:
        ds.dump_raw_dataset(config.folder, config.testing)
    else:
        ds.dump_preprocessed_dataset(config.folder, config.testing, config.separate_seq)

    # if config.dump_text and config.tokenizer_path is None:
    #     ds.save_assets(config.folder)


if __name__ == "__main__":
    dump()
