import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class Analysis:
    def __init__(self, args, loss_object, decoder):
        self.args = args
        self.pos_edge = []
        self.neg_edge = []
        self.loss_object = loss_object
        self.decoder = decoder

    def accumulate(self, pos_edge, neg_edge):
        self.pos_edge.append(pos_edge)
        self.neg_edge.append(neg_edge)

    def analysis_activated_node_indices(self, embeddings, tr_input):
        s_and_t_act_pe_list = []
        s_or_t_act_pe_list = []
        deact_pe_list = []
        s_and_t_act_ne_list = []
        s_or_t_act_ne_list = []
        deact_ne_list = []

        for z, pe, ne, act_ids in zip(
            embeddings, self.pos_edge, self.neg_edge, tr_input["indices_subnodes"]
        ):
            act_s_ids_pe = torch.isin(pe[0], act_ids.to(pe[0]))
            act_t_ids_pe = torch.isin(pe[1], act_ids.to(pe[0]))
            act_s_ids_ne = torch.isin(ne[0], act_ids.to(pe[0]))
            act_t_ids_ne = torch.isin(ne[1], act_ids.to(pe[0]))
            s_and_t_act_pe = pe[:, act_s_ids_pe & act_t_ids_pe]
            s_or_t_act_pe = pe[:, act_s_ids_pe | act_t_ids_pe]
            deact_pe = pe[:, ~(act_s_ids_pe | act_t_ids_pe)]
            s_and_t_act_ne = ne[:, act_s_ids_ne & act_t_ids_ne]
            s_or_t_act_ne = ne[:, act_s_ids_ne | act_t_ids_ne]
            deact_ne = ne[:, ~(act_s_ids_ne | act_t_ids_ne)]

            s_and_t_act_pe_list.append(s_and_t_act_pe)
            s_or_t_act_pe_list.append(s_or_t_act_pe)
            deact_pe_list.append(deact_pe)
            s_and_t_act_ne_list.append(s_and_t_act_ne)
            s_or_t_act_ne_list.append(s_or_t_act_ne)
            deact_ne_list.append(deact_ne)

        s_and_t_act_pe = torch.cat(s_and_t_act_pe_list, dim=1)
        s_or_t_act_pe = torch.cat(s_or_t_act_pe_list, dim=1)
        deact_pe = torch.cat(deact_pe_list, dim=1)
        s_and_t_act_ne = torch.cat(s_and_t_act_ne_list, dim=1)
        s_or_t_act_ne = torch.cat(s_or_t_act_ne_list, dim=1)
        deact_ne = torch.cat(deact_ne_list, dim=1)

        self.args.debug_logger.writer.add_scalar(
            f"Ratio/Ratio_of_s_and_t (#pos/#neg)",
            s_and_t_act_pe.shape[1] / s_and_t_act_ne.shape[1],
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"Ratio/Ratio_of_s_or_t (#pos/#neg)",
            s_or_t_act_pe.shape[1] / s_or_t_act_ne.shape[1],
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"Ratio/Ratio_of_deact (#pos/#neg)",
            deact_pe.shape[1] / deact_ne.shape[1],
            self.args.total_step,
        )

        s_and_t_pe_mse = self.loss_object.mse_from_pred_to_edge(
            z, s_and_t_act_pe, self.decoder, pos_neg="pos"
        )
        s_or_t_pe_mse = self.loss_object.mse_from_pred_to_edge(
            z, s_or_t_act_pe, self.decoder, pos_neg="pos"
        )
        deact_pe_mse = self.loss_object.mse_from_pred_to_edge(
            z, deact_pe, self.decoder, pos_neg="pos"
        )
        s_and_t_ne_mse = self.loss_object.mse_from_pred_to_edge(
            z, s_and_t_act_ne, self.decoder, pos_neg="neg"
        )
        s_or_t_ne_mse = self.loss_object.mse_from_pred_to_edge(
            z, s_or_t_act_ne, self.decoder, pos_neg="neg"
        )
        deact_ne_mse = self.loss_object.mse_from_pred_to_edge(
            z, deact_ne, self.decoder, pos_neg="neg"
        )

        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/1 s_and_t_pe",
            s_and_t_pe_mse,
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/1 s_or_t_pe",
            s_or_t_pe_mse,
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/1 deact_pe",
            deact_pe_mse,
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/2 s_and_t_ne",
            s_and_t_ne_mse,
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/2 s_or_t_ne",
            s_or_t_ne_mse,
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_scalar(
            f"MSE(pred, edge_label)/2 deact_ne",
            deact_ne_mse,
            self.args.total_step,
        )
