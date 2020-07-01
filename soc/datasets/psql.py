from sqlalchemy import create_engine
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Callable


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

    def __init__(
            self,
            no_db: bool = False,
            psql_username: str = 'deepsoc',
            psql_host: str = 'localhost',
            psql_port: int = 5432,
            psql_db_name: str = 'soc'
    ) -> None:
        super(SocPSQLDataset, self).__init__()

        self._length = -1
        self._first_index = 100  # Due to the java implementation

        if no_db:
            self.engine = None
        else:
            self.engine = create_engine(
                'postgresql://{}@{}:{}/{}'.format(
                    psql_username, psql_host, psql_port, psql_db_name
                )
            )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    def _get_states_from_db(self, idx: int) -> pd.DataFrame:
        raise NotImplementedError

    def _get_actions_from_db(self, idx: int) -> pd.DataFrame:
        raise NotImplementedError

    def _get_length(self) -> int:
        raise NotImplementedError

    def get_input_size(self) -> Any:
        raise NotImplementedError

    def get_output_size(self) -> Any:
        raise NotImplementedError

    def get_collate_fn(self) -> Callable:
        raise NotImplementedError
