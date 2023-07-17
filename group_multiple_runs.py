import os
import sys
import numpy as np
import shutil
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from loguru import logger


class GroupMultipleRuns:
    def __init__(self, dpath):
        # preprocess
        for component in os.listdir(dpath):
            if (
                os.path.isdir(os.path.join(dpath, component))
                and not "combined_" in component
            ):
                self.process_each_folder(os.path.join(dpath, component))

    def process_each_folder(self, dpath):
        # produce gathered_events
        assert dpath[0] != "/"

        d = self.tabulate_events(os.path.join(dpath))
        st = dpath.split("/")
        st[-1] = "combined_" + st[-1]

        # delete combined_folder for initialization
        combined_path = os.path.join(*st)
        shutil.rmtree(combined_path, ignore_errors=True)

        # produce mean values
        self.write_combined_events(combined_path, d)

    def tabulate_events(self, dpath):
        summary_iterators = [
            EventAccumulator(os.path.join(dpath, dname)).Reload()
            for dname in os.listdir(dpath)
            if os.path.isdir(os.path.join(dpath, dname))
        ]
        tags = summary_iterators[0].Tags()["scalars"]
        for it in summary_iterators:
            assert it.Tags()["scalars"] == tags
        out = defaultdict(list)
        for tag in tqdm(tags, desc="tabulating events"):
            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                if len(set(e.step for e in events)) != 1:
                    # step 이 sync 가 맞지 않아 평균을 구할 수 없는 상황이면 버림
                    try:
                        del out[tag]
                    except:
                        pass
                    break
                out[tag].append([e.value for e in events])
        return out

    def write_combined_events(self, dpath, d_combined):
        os.makedirs(dpath, exist_ok=True)
        writer = SummaryWriter(dpath)
        tags, tag_values = zip(*d_combined.items())
        logger.info(
            "WARNING: Steps (x-axis) of current function starts from 1. You may need to start from 0."
        )
        for tag, values in tqdm(zip(tags, tag_values), desc="writing combined events"):
            # drop a run which is the most outlier. (100, 3) -> (100, 2) without outlier
            indices = np.argsort(np.array(values).mean(axis=0))[1:]
            means = np.array(values)[:, indices].mean(axis=-1)
            for i, mean in enumerate(means):
                # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=mean)])
                writer.add_scalar(tag, mean, i + 1)
            writer.flush()

    def test(self):
        dpath = "/path/to/root/directory"
        d = self.tabulate_events(dpath)
        self.write_combined_events(dpath, d)


if __name__ == "__main__":
    dpath = sys.argv[1]
    GroupMultipleRuns(dpath)
