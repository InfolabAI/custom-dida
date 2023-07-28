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
- COLLAB and Yelp
    - Download dataset at `./data` from following links
        ```bash
        https://drive.google.com/file/d/19SOqzYEKvkna6DKd74gcJ50Wd4phOHr3/view?usp=share_link
        ```
- BitcoinAlpha, WikiElec and RedditBody
    - Run following commands
        ```bash
        python ./dataset_loader/preprocess_dgl_brw.py # Download dataset, produce DGL graphs and save DGL graphs as pt files to fix node features
        python ./dataset_loader/preprocess_dict_from_dgl.py # Produce dictionaries from DGL graphs, and save dictionaries as pt files becuase we need to fix negative edges for test
        ```

## Usage
- Check args.log_dir in `config.py`
- Run the project
    - **Option 1** - Use `main.py` directly, for example:
        ```bash
        # Our method with node propagation 
        python main.py --model ours --seed 123 --device_id 0 --propagate inneraug --dataset collab --ex_name "aa"

        # DIDA
        python main.py --model dida --seed 123 --device_id 0 --dataset collab --ex_name "Dynamic aug"
        ```
    - **Option 2** - Use `script.sh`
        ```bash
        bash script.sh
        # Then, check the log folder <args.log_dir>/<args.ex_name>/* for this run

        # To check logs, move the log folder to <TENSORBOARD FOLDER> and run tensorboard
        tensorboard --logdir <TENSORBOARD FOLDER> --port <PORT>

        # Check the logs in tensorboard
        ```
- After running the project, if you want to group the results from multiple runs to get the mean values, run following command
    ```bash
    # We assume that <LOG_PATH> has multiple runs with different seeds
    python group_multiple_runs.py <LOG_PATH>

    # Check <LOG_PATH>/combined_<RUN NAME>
    ```

## Paper