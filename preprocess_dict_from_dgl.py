import os
import torch
import dgl
from dataset_loader.template import DatasetTemplate
from model_DIDA.utils.mutils import bi_negative_sampling


# TODO ANKI [OBNOTE: ] - build dict from dgl
class PreprocessDictFromDGL:
    # TODO END ANKI
    def __init__(self, dgl_folder_path, dict_graph_folder_path):
        graph_dict = self.load_pt_files_from_folder(dict_graph_folder_path, torch.load)
        if graph_dict is None:
            graph_dict = self.build_dict_from_dgl(
                dgl_folder_path, dict_graph_folder_path
            )

        self.graph_dict = graph_dict

    def load_pt_files_from_folder(self, folder_path, function):
        """
        주어진 폴더와 하위 폴더에서 .pt 파일을 찾아서 로드하는 함수입니다.

        Args:
            folder_path (str): 탐색할 최상위 폴더의 경로

        Returns:
            List[torch.nn.Module]: 로드한 PyTorch 모델들의 리스트
        """
        graph = None
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".pt"):
                    file_path = os.path.join(root, file_name)
                    graph = function(file_path)
                    break

        return graph

    def build_dict_from_dgl(self, dgl_folder_path, dict_graph_folder_path):
        """
        Load dgl from folder and build dict from it
        """
        graph_dict = {}
        graph = self.load_pt_files_from_folder(
            dgl_folder_path, DatasetTemplate.load_from_path
        )
        # get node features from time step 0 while dropping features from other time stamps
        graph_dict["x"] = graph.graphs[0].ndata["X"]
        graph_dict["train"] = {}
        graph_dict["train"]["pedges"] = []
        for i in range(len(graph.graphs)):
            sources, destinations = graph.graphs[i].edges()
            graph_dict["train"]["pedges"].append(
                torch.concat([sources.unsqueeze(0), destinations.unsqueeze(0)], dim=0)
            )
        graph_dict["train"]["edge_index_list"] = graph_dict["train"]["pedges"]
        graph_dict["train"]["nedges"] = []
        for edges_at_time_t in graph_dict["train"]["pedges"]:
            graph_dict["train"]["nedges"] += [
                bi_negative_sampling(
                    edges_at_time_t,
                    graph_dict["x"].shape[0],
                    graph_dict["x"].shape[0] / 2,
                )
            ]

        try:
            os.makedirs(dict_graph_folder_path)
        except:
            pass
        torch.save(graph_dict, dict_graph_folder_path + "/dict_graph.pt")
        return graph_dict
