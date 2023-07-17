import torch
import numpy as np
import datetime
from adjustText import adjust_text
from matplotlib import pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

HISTOGRAMS_DIR = "./logs/histograms/"


class VisualizeEdges:
    def build_edgedict_from_dynamic_graph(
        self, dynamic_graph_edges: torch.Tensor
    ) -> dict:
        """
        Args:
            dynamic_graph_edges: torch.Tensor
                shape: a list which consists of T tensors where each tensor is [2, num_edges], and T is the number of time steps
                dtype: torch.int64
        """
        edgedict = defaultdict(torch.Tensor)
        with tqdm(enumerate(dynamic_graph_edges)) as pbar:
            pbar.set_description(f"Building edgedict from dynamic graph")
            for t, edges in pbar:
                # [2, num_edges] -> [num_edges, 2]
                edges = edges.t().sort(1)[0]
                for edge in edges:
                    if edgedict[f"{edge[0]}-{edge[1]}"].shape[0] == 0:
                        edgedict[f"{edge[0]}-{edge[1]}"] = torch.zeros(
                            len(dynamic_graph_edges), dtype=torch.int64
                        )
                    # directed graph 라서 여기서 += 를 쓰면 2인 경우도 있음
                    edgedict[f"{edge[0]}-{edge[1]}"][t] += 1
        return edgedict

    def visualize_edges(self, name, edgedict: dict):
        self._visualize_edges_for_each_timestep_matplotlib(name, edgedict)
        self._visualize_consecutive_egdes_matplotlib(name, edgedict)
        self._visualize_number_of_egdes_matplotlib(name, edgedict)
        self._visualize_edges_across_timesteps_matplotlib(name, edgedict)

    def _visualize_edges_for_each_timestep_matplotlib(self, name, edgedict: dict):
        """
        Args:
            the same as visualize_edges_across_timesteps
        """
        list_ = []
        with tqdm(enumerate(edgedict.items())) as pbar:
            pbar.set_description(
                f"Processing _visualize_egdes_for_each_timestep_matplotlib"
            )
            for i, (k, v) in pbar:
                # time step 이 x 축이 되어야 하므로, edge 가 true 인 위치마다 time step 을 추출
                list_.append(np.arange(len(v))[v > 0])
        array_ = np.concatenate(list_, axis=0)
        # plot 1d
        self._plot1d(
            array_,
            "Time step",
            "Probability",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_time_stepsD.png",
            density=True,
        )
        self._plot1d(
            array_,
            "Time step",
            "Number",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_time_steps.png",
            density=False,
        )

    def _visualize_number_of_egdes_matplotlib(self, name, edgedict: dict):
        """
        Args:
            the same as visualize_edges_across_timesteps
        """
        list_ = []
        with tqdm(enumerate(edgedict.items())) as pbar:
            pbar.set_description(f"Processing _visualize_number_of_egdes_matplotlib")
            for i, (k, v) in pbar:
                list_.append((v > 0).sum())
        array_ = np.array(list_)
        xticks = [x for x in range(array_.max() + 1)]

        # plot 1d
        self._plot1d(
            array_,
            "The number of edges across time steps",
            "Probability",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_num_edgesD.png",
            True,
        )
        self._plot1d(
            array_,
            "The number of edges across time steps",
            "Number",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_num_edges.png",
            False,
        )

    def _visualize_consecutive_egdes_matplotlib(self, name, edgedict: dict):
        """
        Args:
            the same as visualize_edges_across_timesteps
        """
        list_ = []
        with tqdm(enumerate(edgedict.items())) as pbar:
            pbar.set_description(f"Processing visualize_consecutive_egdes")
            for i, (k, v) in pbar:
                list_.append(self._calculate_max_consecutive_positive_values(v))
        array_ = np.array(list_)
        xticks = [x for x in range(array_.max() + 1)]

        # plot 1d
        self._plot1d(
            array_,
            "The time lenth of consecutive edges across time steps",
            "Probability",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_consecutive_edgesD.png",
            True,
        )
        self._plot1d(
            array_,
            "The time lenth of consecutive edges across time steps",
            "Number",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_consecutive_edges.png",
            False,
        )

    def _plot1d(self, array_, xlabel, ylabel, savedir, density):
        # set propoerties
        plt.subplots_adjust(left=0.2, bottom=0.2)
        # set xticks
        xticks = [x for x in range(array_.max() + 1)]
        # plot 1d
        n, bins, patches = plt.hist(
            array_,
            bins=[x - 0.5 for x in range(len(xticks) + 1)],
            density=density,
            color="orange",
        )
        ## present probs on each bar
        text_list = []
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height()
            if i % 2 == 0:
                color = "blue"
            else:
                color = "green"
            ### represent probs with .4f format
            if (y == 0) or (len(xticks) > 20):
                continue
            if y >= 1:
                text = f"{y:.0e}".replace("+0", "+")
            elif (y > 0.9) and (y < 1):
                text = f"{y:.1e}".replace("-0", "-")
            elif y <= 0.9:
                text = f"{y:.0e}".replace("-0", "-")
            text_list.append(plt.text(x, y, text, color=color, fontsize=10))
        if len(text_list) > 0:
            adjust_text(
                text_list,
                va="center",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="r", lw=2),
                # only_move={"points": "y", "text": "y", "objects": "y"},
            )
        ## set ticks
        if len(xticks) < 20:
            plt.xticks(xticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(savedir, dpi=300)
        plt.close()

    def _visualize_edges_across_timesteps_matplotlib(self, name, edgedict: dict):
        # build xvalue_to_yedgeid
        xvalue_to_yedgeid = []
        for i, (k, v) in enumerate(edgedict.items()):
            for t in range(v.shape[0]):
                xvalue_to_yedgeid.append([v[t], i])
        xvalue_to_yedgeid = np.array(xvalue_to_yedgeid)

        # set ticks
        logger.info(np.unique(xvalue_to_yedgeid[:, 0]))
        xticks = [x for x in range(np.unique(xvalue_to_yedgeid[:, 0]).max() + 1)]
        yticks = [x for x in range(len(edgedict))]

        # plot 2d
        # plt.rcParams["font.size"] = 15
        plt.hist2d(
            xvalue_to_yedgeid[:, 0],  # xvalues
            xvalue_to_yedgeid[:, 1],  # yedgeids
            bins=[
                [x - 0.5 for x in range(len(xticks) + 1)],
                [y - 0.5 for y in range(len(yticks) + 1)],
            ],
            cmap="Oranges",
            density=True,
        )
        ## set properties
        cb = plt.colorbar()
        cb.set_label("Probability in a bin")
        plt.subplots_adjust(left=0.2, bottom=0.2)
        ## set x ticks
        if len(xticks) < 30:
            plt.xticks(xticks)
        if len(yticks) < 100:
            plt.yticks(yticks)
        plt.xlabel("#edges for each edge id at a time step")
        plt.ylabel("Edge ID")
        plt.title(f"Edges across timesteps")
        ## avoid cutting labels by setting left, bottom margin
        plt.savefig(
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_2d_stats.png",
            dpi=300,
        )
        plt.close()

        # plot 1d
        self._plot1d(
            xvalue_to_yedgeid[:, 0],
            "#edges for each edge id at a time step",
            "Probability",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_statsD.png",
            True,
        )
        self._plot1d(
            xvalue_to_yedgeid[:, 0],
            "#edges for each edge id at a time step",
            "Number",
            f"{HISTOGRAMS_DIR}{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}_1d_stats.png",
            False,
        )

    def _calculate_max_consecutive_positive_values(self, arr):
        max_consecutive_positive_values = 0
        current_consecutive_positive_values = 0

        for num in arr:
            if num >= 1:
                current_consecutive_positive_values += 1
                max_consecutive_positive_values = max(
                    max_consecutive_positive_values, current_consecutive_positive_values
                )
            else:
                current_consecutive_positive_values = 0

        return max_consecutive_positive_values

    def example1(self):
        """
        Examples for visualize_edges_over_timesteps()
        """

        edgedict1 = {
            "0-1": torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).numpy().astype(int),
            "0-2": torch.Tensor([1, 1, 1, 1, 1, 1, 1, 0, 1, 0]).numpy().astype(int),
            "0-3": torch.Tensor([1, 2, 1, 2, 1, 0, 1, 1, 1, 0]).numpy().astype(int),
        }
        edgedict2 = {
            "0-1": torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).numpy().astype(int),
            "0-2": torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 0]).numpy().astype(int),
            "0-3": torch.Tensor([1, 1, 0, 1, 1, 0, 0, 0, 0, 0]).numpy().astype(int),
        }
        edgedict3 = {
            "0-1": torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).numpy().astype(int),
            "0-2": torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).numpy().astype(int),
            "0-3": torch.Tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).numpy().astype(int),
        }
        self.visualize_edges("edgedict1", edgedict1)
        self.visualize_edges("edgedict2", edgedict2)
        self.visualize_edges("edgedict3", edgedict3)

    """
    DEPRECATED
    """

    def visualize_edges_across_timesteps_tensorboard(self, name, edgedict: dict, bins):
        """
        Args:
            edgedict: dict from _build_edgedict_from_dynamic_graph()
        """
        writer = SummaryWriter(
            log_dir=HISTOGRAMS_DIR
            + "/"
            + name
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        with tqdm(enumerate(edgedict.items())) as pbar:
            pbar.set_description(f"Processing visualize_edges_across_timesteps")
            for i, (k, v) in pbar:
                writer.add_histogram(f"{name} bins {bins}", v, i, bins=bins)
        writer.close()

    def visualize_consecutive_egdes_tensorboard(self, name, edgedict: dict, bins):
        """
        Args:
            the same as visualize_edges_across_timesteps
        """
        writer = SummaryWriter(
            log_dir=HISTOGRAMS_DIR
            + "/"
            + name
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        list_ = []
        with tqdm(enumerate(edgedict.items())) as pbar:
            pbar.set_description(f"Processing visualize_consecutive_egdes")
            for i, (k, v) in pbar:
                list_.append(self._calculate_max_consecutive_positive_values(v))

        writer.add_histogram(
            f"consecutive edges of {name} with bins {bins}",
            np.array(list_),
            0,
            bins=bins,
        )
        writer.close()
