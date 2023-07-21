python main.py --model ours --seed 123 --device_id 1 --loguru_level DEBUG --propagate dyaug --dataset yelp --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 123 --device_id 2 --loguru_level DEBUG --propagate dyaug --dataset collab --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 123 --device_id 3 --loguru_level DEBUG --propagate dyaug --dataset bitcoin --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 123 --device_id 4 --loguru_level DEBUG --propagate dyaug --dataset wikielec --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 123 --device_id 6 --loguru_level DEBUG --propagate dyaug --dataset redditbody --ex_name "Comparison of models with edge propagation or without" &

python main.py --model ours --seed 222 --device_id 1 --loguru_level DEBUG --propagate dyaug --dataset yelp --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 2 --loguru_level DEBUG --propagate dyaug --dataset collab --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 3 --loguru_level DEBUG --propagate dyaug --dataset bitcoin --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 5 --loguru_level DEBUG --propagate dyaug --dataset wikielec --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 7 --loguru_level DEBUG --propagate dyaug --dataset redditbody --ex_name "Comparison of models with edge propagation or without" &