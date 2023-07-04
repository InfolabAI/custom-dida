from model_TokenGT.dataset_handler_tokengt_noCD import (
    TokenGTDataset_noCD,
    DatasetConverter_noCD,
)
from model_TokenGT.dataset_handler_tokengt_CD import (
    TokenGTDataset_CD,
    DatasetConverter_CD,
)


def get_data_converter(args):
    if args.model == "tokengt_nocd":
        return TokenGTDataset_noCD, DatasetConverter_noCD
    elif "tokengt_cd" in args.model:
        return TokenGTDataset_CD, DatasetConverter_CD
    elif args.model == "dida":
        return None, None
    else:
        raise NotImplementedError
