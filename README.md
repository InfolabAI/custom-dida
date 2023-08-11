## Dependencies

- CUDA == 11.7
- python == 3.9
    ```shell
    conda create -n py39 python=3.9
    ```
- pytorch == 2.0.1
    ```shell
    pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
- torch-geometric == 2.3.1  **and**  torch-scatter == 2.1.1+pt20cu117
    - Get appropriate `whl` files in [link](https://data.pyg.org/whl/), then run following command
        ```shell
        pip install <scatter NAME>.whl # In this case, from https://data.pyg.org/whl/torch-2.0.1%2Bcu117.html

        pip install <geometric NAME>.whl ...
        ```
    - <font color=#cc0000>**Do not use**</font> `torch-scatter==2.1.1` because `<scatter 2.1.1 NAME>.whl` and `torch-scatter==2.1.1` are differet
- dgl == 1.1.1+cu117
- fairseq == 0.12.2
- gensim == 4.3.1
- tensorboard == 2.13.0
- scikit-learn == 1.3.0
- pandas == 2.0.3
- python-louvain == 0.16
- loguru == 0.7.0
- matplotlib == 3.7.2
- mpmath == 1.3.0
- networkx == 3.1
- numpy == 1.25.1

## Datasets
- Datasets are automatically downloaded and preprocessed

## Usage
- Check `settings/ours-WikiElec-none.json` which means `<method>-<dataset>-<augmentation_method>.json` and defines fixed configurations searched by wandb's hyperparameter optimization (HPO)
- Check `yamls/WikiElec_sweep.yaml` which defines sweep configuration to perform wandb's HPO
- Create wandb's account and initialize your project by following [Quickstart](https://docs.wandb.ai/quickstart)
- Run wandb's sweep by following [Define sweep configuration](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration), [Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps) and [Start sweep agents](https://docs.wandb.ai/guides/sweeps/start-sweep-agents)
- Example
    - run the project on WikiElec without HPO
    ```bash
    python src/main.py --conf_file settings/ours-WikiElec-none.json
    ```

    - run the project on WikiElec with HPO
    ```
    ```bash
    wandb sweep yamls/WikiElect_sweep.yaml
    # wandb's sweep_ID is generated

    # parallel runs accross different terminal windows
    # In terminal window 1
    CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
    # In terminal window 2
    CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
    # In terminal window 3
    CUDA_VISIBLE_DEVICES=2 wandb agent sweep_ID
    ...
    ```
## Paper