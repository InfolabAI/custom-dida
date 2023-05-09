from runner import RunnerProperty


class Trainer:
    def __init__(self, args, model):
        self.runnerProperty = None
        self.args = args
        self.model = model

    def train(self):
        raise NotImplementedError()

    def setRunnerProperty(self, runnerProperty: RunnerProperty):
        self.runnerProperty = runnerProperty
