import argparse
from argparse import ArgumentParser
from sqlalchemy import create_engine
from torch.utils.data import Dataset
from typing import Any
from ..typing import SocConfig, SocCollateFn


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

    _obs_columns = [
        'hexlayout',
        'numberlayout',
        'robberhex',
        'gamestate',
        'devcardsleft',
        'diceresult',
        'startingplayer',
        'currentplayer',
        'playeddevcard',
        'piecesonboard',
        'players',
    ]

    _state_size = [245, 7, 7]
    _action_size = [17, 7, 7]

    def __init__(self, config: SocConfig) -> None:
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

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('no_db', type=str, default=False)
        parser.add_argument('psql_username', type=str, default='deepsoc')
        parser.add_argument('psql_host', type=str, default='localhost')
        parser.add_argument('psql_port', type=int, default=5432)
        parser.add_argument('psql_db_name', type=str, default='soc')
        parser.add_argument('first_index', type=int, default=100)

        return parser

    def _set_props(self, config: SocConfig):
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
