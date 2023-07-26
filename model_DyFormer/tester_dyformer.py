from trainer_and_tester import TrainerAndTester


class TesterDyFormer(TrainerAndTester):
    def __init__(self, args):
        super().__init__(args, None, None)
        pass

    def test(self, epoch, data):
        # Not used
        pass
