# DIDA
## Prepare datasets
```bash
python preprocess_dgl_brw.py # Download dataset, produce DGL graphs and save DGL graphs as pt files to fix node features
python preprocess_dict_from_dgl.py # Produce dictionaries like DIDA datasets from DGL graphs, and save dictionaries as pt files becuase we need to fix negative edges for test

```

## Group multiple runs to get the mean values of multiple runs
```bash
bash script_hee.sh
python group_multiple_runs.py <LOG_PATH>
# check <LOG_PATH>/combined_<EACH RUN>
```