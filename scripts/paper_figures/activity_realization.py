#!/usr/bin/env python
"""Export one title-free paper activity PDF from saved Task 02 data."""
import argparse

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.figures.paper_activity_realizations import plot_paper_activity_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=("tuning", "evaluation"), required=True)
    parser.add_argument("--index", type=int, required=True, help="Zero-based saved realization index.")
    args = parser.parse_args(argv)
    path = plot_paper_activity_figure(load_config(args.config), args.split, args.index)
    print(path)
    return path


if __name__ == "__main__":
    main()
