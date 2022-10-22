import argparse

parser = argparse.ArgumentParser(description="Speech utilities")
subparsers = parser.add_subparsers(required=True, dest="mode")

preprocess_subparser = subparsers.add_parser("preprocess", help="Preprocess a dataset")
preprocess_subparser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="The name of the dataset",
)
preprocess_subparser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="Path to the base dataset directory",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == 'preprocess':
        from preprocessing import datasets

        datasets.processors[args.dataset](args.dataset_dir)