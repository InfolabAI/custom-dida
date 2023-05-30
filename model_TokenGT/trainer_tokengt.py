from trainer import Trainer
from model_DIDA.utils.inits import prepare
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
from hook import gradient_hook, forward_hook
import time
from tqdm import tqdm


class Trainer_TokenGT(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        pass

    def train(self, epoch, data):
        super().train(data)
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        args = self.runnerProperty.args
        self.model.train()
        optimizer = self.runnerProperty.optimizer

        # TODO ANKI [OBNOTE: ] - 모든 레이어에 register hook 설정
        # 모든 레이어에 register hook 설정
        # for module in self.model.modules():
        #    module.register_backward_hook(gradient_hook)
        #    module.register_forward_hook(forward_hook)
        # TODO END ANKI

        # train
        edge_index = []
        edge_label = []
        epoch_losses = []
        ## edge label construction
        # with tqdm(total=self.runnerProperty.len_train - 1) as pbar:
        for t in range(self.runnerProperty.len_train - 1):
            print(f"t = {t}")
            # pbar.set_description(f"Processing item {t}")

            batch = data[t]
            z = self.model(batch)
            # catch nan and inf
            # TODO ANKI [OBNOTE: ] - catch nan or inf
            # if torch.isnan(z.node_data).any() or torch.isinf(z.node_data).any():
            #    print("   nan or inf")
            #    breakpoint()
            # TODO END ANKI

            pos_edge_index = prepare(data.data, t + 1)[0]
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) * args.sampling_times,
                )
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
            edge_label = torch.cat([pos_y, neg_y], dim=0)

            criterion = torch.nn.BCELoss()

            def cal_loss(y, label):
                return criterion(y, label)

            cy = self.model.cs_decoder(z, edge_index)

            loss = cal_loss(cy, edge_label)

            st = time.process_time()
            optimizer.zero_grad()
            loss.backward()
            # TODO ANKI [OBNOTE: ] - grad_fn 을 따라 올라가면서 연산과정을 확인하는 방법
            # grad_fn = loss.grad_fn
            # while grad_fn is not None:
            #    print(grad_fn)
            #    grad_fn = (
            #        grad_fn.next_functions[0][0] if grad_fn.next_functions else None
            #    )
            # TODO END ANKI
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-5)
            optimizer.step()
            epoch_losses.append(loss.detach().item())
            # pbar.set_postfix(loss=f"{loss.detach().item():.4f}")
            # pbar.update(1)
            # print(loss)
            # TODO ANKI [OBNOTE: ] - https://discuss.pytorch.org/t/gradient-value-is-nan/91663/3 https://github.com/Lightning-AI/lightning/issues/9799#issuecomment-932980107(do not use it in the beginning of the epochs)
            # torch.autograd.set_detect_anomaly(True)
            # TODO END ANKI

        average_epoch_loss = np.array(epoch_losses).mean()

        return average_epoch_loss, [], [], []
