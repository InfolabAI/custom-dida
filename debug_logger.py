from loguru import logger
from collections import defaultdict


class DebugLogger:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.count_log = defaultdict(lambda: 0)

    def is_log(self, name, interval, additional_cond=True):
        cond1 = self.args.loguru_level == "DEBUG"
        cond2 = self.args.total_step == 0 or self.args.total_step % interval == 0
        cond = (cond1 and cond2 and additional_cond) or (self.count_log[name] == 0)
        if cond:
            self.count_log[name] += 1
        return cond

    def histogram(self, name, tensor, interval, additional_cond=True):
        if self.is_log(name, interval, additional_cond):
            self.writer.add_histogram(name, tensor, self.args.total_step)

    def scalar(self, name, scalar, interval, additional_cond=True):
        if self.is_log(name, interval, additional_cond):
            self.writer.add_scalar(name, scalar, self.args.total_step)

    def loguru(self, name, text, interval, additional_cond=True):
        text = str(text)
        if self.is_log(name, interval, additional_cond):
            logger.debug(f"{self.args.total_step} : {name} : {text}")
