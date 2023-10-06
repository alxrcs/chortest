import pandas as pd
from glob import glob
from pathlib import Path
import sys

ALL_STR = "all"


def main():

    assert len(sys.argv) > 1, "You should specify a path for the experiment"
    p = Path(sys.argv[1])
    assert p.exists() and p.is_dir()

    dfs = [
        pd.read_csv(p, index_col=0)
        for p in glob(f"{p}/logs_mut_stats_*.csv")
        if not ALL_STR in str(p)
    ]

    df = pd.concat(dfs)

    df = df.reset_index().drop("index", axis=1)
    df.to_csv(f"{str(p)}/logs_mut_stats_{ALL_STR}.csv")


if __name__ == "__main__":
    main()
