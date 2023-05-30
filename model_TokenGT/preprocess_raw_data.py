import torch
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm


class PreprocessBitcoinAlpha:
    def __init__(self):
        self.dir_ = "./raw_data/BitcoinAlpha/"

    def preprocess_BitcoinAlpha(self):
        # q: what is header in read_csv?
        # a: header is the row number to use as the column names, and the start of the data.
        column_names = ["source", "target", "rating", "time"]
        df = pd.read_csv(
            self.dir_ + "soc-sign-bitcoinalpha.csv", header=None, names=column_names
        )
        # NOTE we ignore day, hour, minute, second to reduce the number of time steps
        df.time = df.time.apply(
            lambda x: datetime.datetime.fromtimestamp(x).replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
            )
        )
        return df

    def retrieve_edges(self):
        df = self.preprocess_BitcoinAlpha()
        # retrieve edges by sorted time
        edges_across_time = []
        print(
            f"The number of time steps is {df.time.unique().shape} and an example of time steps is {df.time[0]}"
        )
        with tqdm(np.sort(df.time.unique())) as pbar:
            pbar.set_description("Preprocessing raw data to retrieve edges")
            for t in pbar:
                df_t = df[df.time == t]
                src_nodes = (
                    torch.Tensor(df_t.source.to_numpy()).to(torch.long).unsqueeze(0)
                )
                dst_nodes = (
                    torch.Tensor(df_t.target.to_numpy()).to(torch.long).unsqueeze(0)
                )
                edges_t = torch.concat([src_nodes, dst_nodes], dim=0)
                edges_across_time.append(edges_t)

        return edges_across_time


if __name__ == "__main__":
    pr_BitcoinAlpha = PreprocessBitcoinAlpha()
    edges_across_time = pr_BitcoinAlpha.retrieve_edges()
    breakpoint()
    print()
