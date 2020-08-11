import os
import time
import shutil
import unittest
import pandas as pd
from unittest.mock import MagicMock
from pytorch_lightning import seed_everything, Trainer
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from soc import models, datasets
from soc.training import Runner, SocConfig
from soc.datasets import make_dataset

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')

_DATASET_PATH = os.path.join(fixture_dir, 'soc_5_fullseq.pt')


class TestTraining(unittest.TestCase):

    df_states: pd.DataFrame
    df_actions: pd.DataFrame

    obs_files = [
        os.path.join(fixture_dir, 'small_obsgamestates_100.csv'),
        os.path.join(fixture_dir, 'small_obsgamestates_101.csv'),
    ]
    actions_files = [
        os.path.join(fixture_dir, 'small_gameactions_100.csv'),
        os.path.join(fixture_dir, 'small_gameactions_101.csv'),
    ]

    _get_states_from_db_se_f = None
    _get_actions_from_db_se_f = None

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

        cls.states = [pd.read_csv(file) for file in cls.obs_files]
        cls.actions = [pd.read_csv(file) for file in cls.actions_files]

    def setUp(self):
        self.folder = os.path.join(fixture_dir, str(int(time.time() * 100000000)))

    def tearDown(self):
        return shutil.rmtree(self.folder)

    def test_training_socseqsas_convlstm(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return actions[idx]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=2)

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=["generic/model=convlstm", "generic/dataset=psqlseqsatos"]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_socseqsas_conv3dmodel(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return actions[idx]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=2)

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=["generic/model=conv3d", "generic/dataset=psqlseqsatos"]
            )
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_socforward_resnet(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=2)
                dataset._get_nb_steps = MagicMock(return_value=[9, 9])

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=["generic/model=resnet18", "generic/dataset=preprocessedforwardsatosa"]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_socforward_resnet_policy(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=4)
                dataset._get_nb_steps = MagicMock(return_value=[9, 9, 9, 9])

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=resnet18policy",
                    "generic/dataset=preprocessedforwardsatosapolicy"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_socseq_conv3d_policy(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=4)
                dataset._get_nb_steps = MagicMock(return_value=[9, 9, 9, 9])

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=conv3dpolicy", "generic/dataset=preprocessedseqsatosapolicy"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

    def test_training_socseq_convlstm_policy(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        class TestRunner(Runner):
            def setup_dataset(self):
                dataset = make_dataset(self.hparams.dataset)
                dataset._get_states_from_db = MagicMock(side_effect=_get_states_from_db_se_f)
                dataset._get_actions_from_db = MagicMock(side_effect=_get_actions_from_db_se_f)
                dataset._get_length = MagicMock(return_value=4)
                dataset._get_nb_steps = MagicMock(return_value=[9, 9, 9, 9])

                return dataset

        with initialize(config_path=os.path.join(".", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "generic/model=convlstmpolicy", "generic/dataset=preprocessedseqsatosapolicy"
                ]
            )
            config.generic.dataset.dataset_path = _DATASET_PATH
            config.trainer.default_root_dir = self.folder

            seed_everything(config['generic']['seed'])
            runner = TestRunner(config['generic'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)
