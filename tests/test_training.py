import os
import time
import shutil
import unittest
import pandas as pd
import torch
from unittest.mock import MagicMock
from pytorch_lightning import seed_everything, Trainer
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from soc import models, datasets
from soc.training import SocConfig
from soc.datasets import make_dataset
from soc.runners import make_runner

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')

_DATASET_PATH = os.path.join(fixture_dir, 'soc_seq_3_fullseq.pt')
_RAW_DATASET_PATH = os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt')
_TEXT_BERT_DATASET_PATH = os.path.join(fixture_dir, 'soc_text_bert_3_fullseq.pt')
_RAW_TEXT_BERT_DATASET_PATH = os.path.join(fixture_dir, 'soc_text_bert_3_raw_df.pt')


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=SocConfig)
        cs.store(group="generic/model", name="convlstm", node=models.ConvLSTMConfig)
        cs.store(group="generic/model", name="convlstmpolicy", node=models.ConvLSTMConfig)
        cs.store(group="generic/model", name="conv3d", node=models.Conv3dModelConfig)
        cs.store(group="generic/model", name="conv3dpolicy", node=models.Conv3dModelConfig)
        cs.store(group="generic/model", name="resnet18", node=models.ResNetConfig)
        cs.store(group="generic/model", name="resnet18policy", node=models.ResNetConfig)
        cs.store(group="generic/model", name="resnet18fusionpolicy", node=models.ResNetFusionConfig)
        cs.store(
            group="generic/model", name="resnet18meanconcatpolicy", node=models.ResNetFusionConfig
        )
        cs.store(group="generic/dataset", name="psqlseqsatos", node=datasets.PSQLConfig)
        cs.store(
            group="generic/dataset",
            name="preprocessedforwardsatosa",
            node=datasets.PreprocessedForwardConfig
        )
        cs.store(
            group="generic/dataset",
            name="preprocessedforwardsatosapolicy",
            node=datasets.PreprocessedForwardConfig
        )
        cs.store(
            group="generic/dataset",
            name="preprocessedseqsatosapolicy",
            node=datasets.PreprocessedSeqConfig
        )
        cs.store(
            group="generic/dataset",
            name="psqltextbertforwardsatosapolicy",
            node=datasets.PSQLTextForwardConfig
        )
        cs.store(
            group="generic/dataset",
            name="preprocessedtextbertforwardsatosapolicy",
            node=datasets.PreprocessedTextForwardConfig
        )
        cs.store(
            group="generic/dataset",
            name="filetextbertforwardsatosapolicy",
            node=datasets.FileTextForwardConfig
        )

        cls.data = torch.load(_RAW_DATASET_PATH)
        cls.data_text_bert = torch.load(_RAW_TEXT_BERT_DATASET_PATH)

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return cls.data[idx][0]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return cls.data[idx][1]

        def _get_length_se_f() -> int:
            return len(cls.data)

        def setup_dataset(self, hparams):
            dataset = make_dataset(hparams.dataset)
            dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
            dataset._get_length = MagicMock(side_effect=_get_length_se_f)

            return dataset, None

        cls.setup_dataset = setup_dataset

        def _get_text_states_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = cls.data_text_bert[table_id][0]
            return seq[start_row_id:end_row_id]

        def _get_text_actions_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = cls.data_text_bert[table_id][1]
            return seq[start_row_id:end_row_id]

        def _get_text_chats_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = cls.data_text_bert[table_id][2]
            return seq[start_row_id:end_row_id]

        def _get_text_nb_steps_se_f():
            return [len(cls.data_text_bert[i][0]) for i in range(len(cls.data_text_bert))]

        def _get_text_length_se_f() -> int:
            return len(cls.data_text_bert)

        def setup_text_dataset(self, hparams):
            dataset = make_dataset(hparams.dataset)
            dataset._get_states_from_db = MagicMock(side_effect=_get_text_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=_get_text_actions_from_db_se_f)
            dataset._get_chats_from_db = MagicMock(side_effect=_get_text_chats_from_db_se_f)
            dataset._get_nb_steps = MagicMock(side_effect=_get_text_nb_steps_se_f)
            dataset._get_length = MagicMock(side_effect=_get_text_length_se_f)

            return dataset, None

        cls.setup_text_dataset = setup_text_dataset

    def setUp(self):
        self.folder = os.path.join(fixture_dir, str(int(time.time() * 100000000)))

    def tearDown(self):
        if os.path.isdir(self.folder):
            shutil.rmtree(self.folder)

    def test_training_soc_psql_seq_sas_convlstm(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=convlstm",
                    "generic/dataset=psqlseqsatos",
                    "generic.runner_name=SOCSupervisedSeqRunner"
                ]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            runner.setup_dataset = self.setup_dataset
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_psql_seq_sas_conv3d(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=conv3d",
                    "generic/dataset=psqlseqsatos",
                    "generic.runner_name=SOCSupervisedSeqRunner"
                ]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            runner.setup_dataset = self.setup_dataset
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_preprocessed_seq_conv3dpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=conv3dpolicy",
                    "generic/dataset=preprocessedseqsatosapolicy",
                    "generic.runner_name=SOCSeqPolicyRunner"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_preprocessed_seq_convlstmpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=convlstmpolicy",
                    "generic/dataset=preprocessedseqsatosapolicy",
                    "generic.runner_name=SOCSeqPolicyRunner"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_preprocessed_forward_resnet(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18",
                    "generic/dataset=preprocessedforwardsatosa",
                    "generic.runner_name=SOCSupervisedForwardRunner"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_preprocessed_forward_resnetpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18policy",
                    "generic/dataset=preprocessedforwardsatosapolicy",
                    "generic.runner_name=SOCForwardPolicyRunner"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_psql_forward_resnetfusionpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18fusionpolicy",
                    "generic/dataset=psqltextbertforwardsatosapolicy",
                    "generic.runner_name=SOCTextForwardPolicyRunner"
                ]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            runner.setup_dataset = self.setup_text_dataset
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_preprocessed_forward_resnetfusionpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18fusionpolicy",
                    "generic/dataset=preprocessedtextbertforwardsatosapolicy",
                    "generic.runner_name=SOCTextForwardPolicyRunner"
                ]
            )
            config.generic.dataset.dataset_path = _TEXT_BERT_DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_psql_forward_resnetmeanconcatpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18meanconcatpolicy",
                    "generic/dataset=psqltextbertforwardsatosapolicy",
                    "generic.runner_name=SOCTextForwardPolicyRunner"
                ]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            runner.setup_dataset = self.setup_text_dataset
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_soc_file_forward_resnetmeanconcatpolicy(self):
        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18meanconcatpolicy",
                    "generic/dataset=filetextbertforwardsatosapolicy",
                    "generic.runner_name=SOCTextForwardPolicyRunner"
                ]
            )
            config.generic.dataset.dataset_path = _RAW_TEXT_BERT_DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = make_runner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)
