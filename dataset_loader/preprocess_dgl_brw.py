from dataset_loader.link import RedditBody
from dataset_loader.link import WikiElec

from dataset_loader.link import BitcoinAlpha

btdt = BitcoinAlpha(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
)
btdt.preprocess()
rbdt = RedditBody(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
)
rbdt.preprocess()
wedt = WikiElec(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
)
wedt.preprocess()
