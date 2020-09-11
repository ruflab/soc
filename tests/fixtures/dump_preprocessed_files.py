import os
from os.path import isfile, join
import soc
from dataclasses import dataclass
import hydra
from typing import Optional
# from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

fixture_dir = os.path.dirname(os.path.realpath(__file__))


@dataclass
class DumpConfig:
    dataset_path: str = ''
    folder: str = fixture_dir

    raw: bool = False
    testing: bool = True
    separate_seq: bool = False

    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=DumpConfig)


@hydra.main(config_name='config')
def dump(config):
    onlyfiles = [f for f in os.listdir(fixture_dir) if isfile(join(fixture_dir, f))]
    for file in onlyfiles:
        if 'df.pt' in file:
            config.dataset_path = os.path.join(fixture_dir, file)
            if '_text_' in file:
                ds = soc.datasets.SocFileTextBertSeqDataset(config)
            else:
                ds = soc.datasets.SocFileSeqDataset(config)
            ds.dump_preprocessed_dataset(config.folder, config.testing, config.separate_seq)


if __name__ == "__main__":
    dump()
