import os
import soc
import argparse
from soc.datasets import PSQLConfig

cfd = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cfd, '..', 'data')
fixture_dir = os.path.join(cfd, '..', 'tests', 'fixtures')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')

    parser.add_argument('--testing', type=bool, default=False, help='Should we dump testing')

    args, _ = parser.parse_known_args()

    config = PSQLConfig()
    if args.testing is True:
        ds = soc.datasets.SocPSQLSeqDataset(config)
        ds.dump_preprocessed_dataset(fixture_dir, True)
    else:
        ds = soc.datasets.SocPSQLSeqDataset(config)
        ds.dump_preprocessed_dataset(data_folder)
