from model_TokenGT.dataset_handler_tokengt_noCD import (
    TokenGTDataset_noCD,
    DatasetConverter,
)
from model_TokenGT.dataset_handler_tokengt_CD import (
    TokenGTDatasetCD,
    DatasetConverterCD,
)


def get_data_converter(args):
    if args.model == "tokengt_nocd":
        return TokenGTDataset_noCD, DatasetConverter
    elif "tokengt_cd" in args.model:
        return TokenGTDatasetCD, DatasetConverterCD
    elif args.model == "dida":
        return None, None
    else:
        raise NotImplementedError
