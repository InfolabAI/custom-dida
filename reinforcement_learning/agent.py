import numpy as np
import math
import torch


class RLAgent:
    def __init__(self, env, actor, args):
        self.actor = actor
        self.env = env
        self.args = args

        self.total_steps = 0
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train_episode(self):
        self.__set_train()
        states = self.env.reset()

        actions = self.actor(states, deterministic=False)

        # ror = torch.from_numpy(self.env.ror).to(self.args.device)
        # normed_ror = (ror - torch.mean(ror, dim=-1, keepdim=True)) / (
        #    torch.std(ror, dim=-1, keepdim=True) + EPS
        # )

        rewards = self.env.step(actions)

        asu_grad = torch.sum(normed_ror * scores_p, dim=-1)
        asu_grad = asu_grad.clamp(min=0 + EPS)
        steps_asu_grad.append(torch.log(asu_grad))

        gradient_asu = torch.stack(steps_asu_grad, dim=1)
        loss = -(gradient_asu)
        loss = loss.mean()
        assert not torch.isnan(loss)
        self.optimizer.zero_grad()
        loss = loss.contiguous()
        loss.backward()
        grad_norm, grad_norm_clip = self.clip_grad_norms(
            self.optimizer.param_groups, self.args.max_grad_norm
        )
        self.optimizer.step()

        rtns = (agent_wealth[:, -1] / agent_wealth[:, 0]).mean()
        avg_rho = np.mean(rho_records)
        avg_mdd = mdd.mean()

        return rtns, avg_rho, avg_mdd

    def evaluation(self, date_data=None, logger=None):
        self.__set_test()
        states, masks = self.env.reset()
        while True:
            steps += 1
            cursor_list.append(int(self.env.src.data_cursor))
            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None

            weights, rho, _, _ = self.actor(x_a, x_m, masks, deterministic=True)

            next_states, rewards, _, masks, done, info = self.env.step(
                weights, rho.detach().cpu().numpy()
            )

            agent_wealth = np.concatenate(
                (agent_wealth, info["total_value"][..., None]), axis=-1
            )

            weights_list.append(weights)
            rho_list.append(float(rho.detach().cpu().numpy()))
            print(
                f"long position - {np.arange(self.args.num_assets)[(np.squeeze(weights, 0)[:self.args.num_assets]>0)]} short position - {np.arange(self.args.num_assets)[(np.squeeze(weights, 0)[self.args.num_assets:]>0)]} rho - {float(rho.detach().cpu().numpy())} wealth - {info['total_value']}\n"
            )
            states = next_states

            if done:
                break

        return agent_wealth, weights_list, cursor_list, rho_list

    def clip_grad_norms(self, param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm
        :param param_groups:
        :param max_norm:
        :return: gradient norms before clipping
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group["params"],
                max_norm
                if max_norm > 0
                else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2,
            )
            for group in param_groups
        ]
        grad_norms_clipped = (
            [min(g_norm, max_norm) for g_norm in grad_norms]
            if max_norm > 0
            else grad_norms
        )
        return grad_norms, grad_norms_clipped

    def __set_train(self):
        self.actor.train()
        self.env.set_train()

    def __set_eval(self):
        self.actor.eval()
        self.env.set_eval()

    def __set_test(self):
        self.actor.eval()
        self.env.set_test()

    def __set_test_train(self):
        self.actor.eval()
        self.env.set_train()
