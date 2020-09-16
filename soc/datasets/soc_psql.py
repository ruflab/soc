from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import Any


@dataclass
class PSQLConfig:
    name: str = MISSING
    no_db: bool = False
    psql_username: str = 'deepsoc'
    psql_host: str = 'localhost'
    psql_port: int = 5432
    psql_db_name: str = 'soc'
    psql_password: str = MISSING

    first_index: int = 100  # Due to the java implementation
    shuffle: bool = True


class SocPSQLDataset(Dataset):
    """
        Defines a Settlers of Catan postgresql dataset.

        Args:
            psql_username: (str) username
            psql_host: (str) host
            psql_port: (int) port
            psql_db_name: (str) database name

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """
    def __init__(self, omegaConf: DictConfig) -> None:
        super(SocPSQLDataset, self).__init__()

        self.no_db = omegaConf['no_db']
        self.psql_username = omegaConf['psql_username']
        self.psql_host = omegaConf['psql_host']
        self.psql_port = omegaConf['psql_port']
        self.psql_db_name = omegaConf['psql_db_name']
        self.psql_password = omegaConf['psql_password']

        self._length = -1
        self._first_index = omegaConf['first_index']

        if self.no_db:
            self.engine = None
        else:
            # We are not using a pool of connections
            # because it does not work well with multiprocessing
            # see https://stackoverflow.com/questions/41279157/connection-problems-with-sqlalchemy-and-multiple-processes  # noqa
            # TODO: Find a way to use a pool with multiprocessing
            self.engine = create_engine(
                'postgresql://{}:{}@{}:{}/{}'.format(
                    self.psql_username,
                    self.psql_password,
                    self.psql_host,
                    self.psql_port,
                    self.psql_db_name
                ),
                poolclass=NullPool
            )

        self._set_props(omegaConf)

    def _set_props(self, config):
        pass

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    def get_input_size(self):
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self):
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self):
        return None

    def get_output_metadata(self):
        raise NotImplementedError
