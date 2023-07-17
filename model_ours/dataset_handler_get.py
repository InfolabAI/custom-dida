from model_ours.dataset_handler import DatasetConverter_CD


def get_data_converter(args):
    if args.model == "ours":
        return DatasetConverter_CD
    elif args.model == "dida":
        return None
    else:
        raise NotImplementedError
