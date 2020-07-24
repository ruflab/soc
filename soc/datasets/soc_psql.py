from sqlalchemy import create_engine
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import Any
from ..typing import SocCollateFn


@dataclass
class PSQLConfig(DictConfig):
    name: str = MISSING
    no_db: bool = False
    psql_username: str = 'deepsoc'
    psql_host: str = 'localhost'
    psql_port: int = 5432
    psql_db_name: str = 'soc'

    first_index: int = 100
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

    _length: int

    def __init__(self, config: PSQLConfig) -> None:
        super(SocPSQLDataset, self).__init__()

        self.no_db = config.get('no_db', False)
        self.psql_username = config.get('psql_username', 'deepsoc')
        self.psql_host = config.get('psql_host', 'localhost')
        self.psql_port = config.get('psql_port', 5432)
        self.psql_db_name = config.get('psql_db_name', 'soc')

        self._length = -1
        self._first_index = config.get('first_index', 100)  # Due to the java implementation

        if self.no_db:
            self.engine = None
        else:
            self.engine = create_engine(
                'postgresql://{}@{}:{}/{}'.format(
                    self.psql_username, self.psql_host, self.psql_port, self.psql_db_name
                )
            )

        self._set_props(config)

    def _set_props(self, config):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    def get_input_size(self) -> Any:
        raise NotImplementedError

    def get_output_size(self) -> Any:
        raise NotImplementedError

    def get_collate_fn(self) -> SocCollateFn:
        raise NotImplementedError
