from sqlalchemy import create_engine
from torch.utils.data import Dataset
import pandas as pd


class SocPSQLDataset(Dataset):
    """
        Defines a Settlers of Catan postgresql dataset.

        Args:
            df: pd.DataFrame

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    _length: int

    def __init__(
        self,
        psql_username: str = 'deepsoc',
        psql_host: str = 'localhost',
        psql_port: int = 5432,
        psql_db_name: str = 'soc'
    ) -> None:
        super(SocPSQLDataset, self).__init__()

        self.engine = create_engine(
            'postgresql://{}@{}:{}/{}'.format(psql_username, psql_host, psql_port, psql_db_name)
        )

    def __len__(self) -> int:
        return self._get_length()

    def __getitem__(self, idx: int):
        df_states = self._get_states_from_db(idx)
        df_actions = self._get_actions_from_db(idx)

        # TODO:
        # Merge columns into 1 feature vector
        # Return a sequence of 2d arrays seq_length x (state, action)
        seq = (df_states, df_actions)

        return seq

    def _get_states_from_db(self, idx: int) -> pd.DataFrame:
        db_id = 100 + idx  # The first row in the DB starts at 100 in the JAVA app

        query = "SELECT * from obsgamestates_{}".format(db_id)
        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_actions_from_db(self, idx: int) -> pd.DataFrame:
        query = "SELECT * from gameactions_{}".format(idx)
        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_length(self) -> int:
        if self._length is None:
            query = r"SELECT count(tablename) FROM pg_catalog.pg_tables WHERE \
                schemaname != 'pg_catalog' AND \
                schemaname != 'information_schema' AND \
                tablename SIMILAR TO 'obsgamestates_%\d\d\d+%'"

            self._length = self.engine.execute(query)

        return self._length
