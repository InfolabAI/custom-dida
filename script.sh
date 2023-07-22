# 3.1G
python main.py --model ours --seed 123 --device_id 0 --propagate no --dataset yelp --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
# 3.1G
python main.py --model ours --seed 123 --device_id 1 --propagate no --dataset collab --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
# 4.5G
python main.py --model ours --seed 123 --device_id 2 --propagate no --dataset bitcoin --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
# 6.6G
python main.py --model ours --seed 123 --device_id 3 --propagate no --dataset wikielec --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
# 10G
python main.py --model ours --seed 123 --device_id 4 --propagate no --dataset redditbody --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &

python main.py --model ours --seed 222 --device_id 0 --propagate no --dataset yelp --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
python main.py --model ours --seed 222 --device_id 1 --propagate no --dataset collab --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
python main.py --model ours --seed 222 --device_id 5 --propagate no --dataset bitcoin --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
python main.py --model ours --seed 222 --device_id 6 --propagate no --dataset wikielec --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
python main.py --model ours --seed 222 --device_id 7 --propagate no --dataset redditbody --ex_name "Comparison of models with edge propagation or without/epoch_update_dropout" &
