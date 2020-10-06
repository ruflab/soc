import os
import time
import shutil
import unittest
import torch
from pytorch_lightning import seed_everything, Trainer
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from soc import models, datasets
from soc.training import SocConfig
from soc.runners import make_runner

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')

_DATASET_PATH = os.path.join(fixture_dir, 'soc_seq_3_fullseq.pt')
_RAW_DATASET_PATH = os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt')
_TEXT_BERT_DATASET_PATH = os.path.join(fixture_dir, 'soc_text_bert_3_fullseq.pt')
_RAW_TEXT_BERT_DATASET_PATH = os.path.join(fixture_dir, 'soc_text_bert_3_raw_df.pt')


class TestTextPolicyRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=SocConfig)
        cs.store(group="runner/model", name="convlstm", node=models.ConvLSTMConfig)
        cs.store(group="runner/model", name="convlstmpolicy", node=models.ConvLSTMConfig)
        cs.store(group="runner/model", name="conv3d", node=models.Conv3dModelConfig)
        cs.store(group="runner/model", name="conv3dpolicy", node=models.Conv3dModelConfig)
        cs.store(group="runner/model", name="resnet18", node=models.ResNetConfig)
        cs.store(group="runner/model", name="resnet18policy", node=models.ResNetConfig)
        cs.store(group="runner/model", name="resnet18fusionpolicy", node=models.ResNetFusionConfig)
        cs.store(group="runner/dataset", name="psqlseqsatos", node=datasets.PSQLConfig)
        cs.store(
            group="runner/dataset",
            name="preprocessedforwardsatosa",
            node=datasets.PreprocessedForwardConfig
        )
        cs.store(
            group="runner/dataset",
            name="preprocessedforwardsatosapolicy",
            node=datasets.PreprocessedForwardConfig
        )
        cs.store(
            group="runner/dataset",
            name="preprocessedseqsatosapolicy",
            node=datasets.PreprocessedSeqConfig
        )
        cs.store(
            group="runner/dataset",
            name="psqltextbertforwardsatosapolicy",
            node=datasets.PSQLTextForwardConfig
        )
        cs.store(
            group="runner/val_dataset",
            name="filetextbertforwardsatosapolicy",
            node=datasets.FileTextForwardConfig
        )
        cs.store(
            group="runner/dataset",
            name="filetextbertforwardsatosapolicy",
            node=datasets.FileTextForwardConfig
        )

    def setUp(self):
        self.folder = os.path.join(fixture_dir, str(int(time.time() * 100000000)))

    def tearDown(self):
        if os.path.isdir(self.folder):
            shutil.rmtree(self.folder)

    def test_training_only_cnn_weights(self):
        with initialize(config_path=os.path.join("..", "fixtures", "conf")):
            config = compose(
                config_name="config",
                overrides=[
                    "runner/model=resnet18fusionpolicy",
                    "runner/dataset=filetextbertforwardsatosapolicy",
                    "runner/val_dataset=filetextbertforwardsatosapolicy",
                    "runner.runner_name=SOCTextForwardPolicyRunner"
                ]
            )
            config.runner.dataset.dataset_path = _RAW_TEXT_BERT_DATASET_PATH
            config.runner.val_dataset.dataset_path = _RAW_TEXT_BERT_DATASET_PATH
            config.runner.train_cnn = True
            config.runner.train_fusion = False
            config.runner.train_heads = True

            config.trainer.default_root_dir = self.folder
            config.trainer.fast_dev_run = False

            # We rely on seeds to copy the init weights
            seed_everything(config['runner']['seed'])
            r_copy = make_runner(config['runner'])
            r_copy.setup('fit')

            seed_everything(config['runner']['seed'])
            runner = make_runner(config['runner'])
            trainer = Trainer(**config['trainer'], deterministic=True)
            trainer.fit(runner)

            zipped_params = zip(r_copy.model.fusion.parameters(), runner.model.fusion.parameters())
            for param_copy, param in zipped_params:
                assert torch.all(torch.eq(param_copy, param))

            zipped_params = zip(
                r_copy.model.spatial_state_head.parameters(),
                runner.model.spatial_state_head.parameters()
            )
            for param_copy, param in zipped_params:
                assert not torch.all(torch.eq(param_copy, param))

            zipped_params = zip(
                r_copy.model.linear_state_head.parameters(),
                runner.model.linear_state_head.parameters()
            )
            for param_copy, param in zipped_params:
                assert not torch.all(torch.eq(param_copy, param))

            # zipped_params = zip(
            #     r_copy.model.policy_head.parameters(), runner.model.policy_head.parameters()
            # )
            # for param_copy, param in zipped_params:
            #     assert not torch.all(torch.eq(param_copy, param))

            zipped_params = zip(r_copy.model.cnn.parameters(), runner.model.cnn.parameters())
            for param_copy, param in zipped_params:
                assert not torch.all(torch.eq(param_copy, param))
                break  # Not all layers are learnable so we check only the first one
