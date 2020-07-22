import os
import soc
import argparse

cfd = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(cfd, '..', 'data')
fixture_dir = os.path.join(cfd, '..', 'tests', 'fixtures')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')

    parser.add_argument('--testing', type=bool, default=False, help='Should we dump testing')

    args, _ = parser.parse_known_args()

    if args.testing is True:
        ds = soc.datasets.SocPSQLSeqDataset({})
        ds.dump_preprocessed_dataset(fixture_dir, 5)
    else:
        ds = soc.datasets.SocPSQLSeqDataset({})
        ds.dump_preprocessed_dataset(data_folder)
