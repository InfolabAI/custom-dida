from link import RedditBody
from link import WikiElec
from link import BitcoinAlpha

# init 에서 이미 process() 까지 모두 실행함
btdt = BitcoinAlpha(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
    seed=117,
)
rbdt = RedditBody(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
    seed=117,
)
wedt = WikiElec(
    input_dim=32,
    train_ratio=0.7,
    val_ratio=0.1,
    device="cpu",
    data_dir="raw_data",
    time_aggregation=3600 * 24 * 7 * 2,
    seed=117,
)
