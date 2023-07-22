# 3.1G
python main.py --model ours --seed 123 --device_id 0 --propagate inneraug --dataset yelp --ex_name "Comparison of models with edge propagation or without" &
# 3.1G
python main.py --model ours --seed 123 --device_id 1 --propagate inneraug --dataset collab --ex_name "Comparison of models with edge propagation or without" &
# 4.5G
python main.py --model ours --seed 123 --device_id 2 --propagate inneraug --dataset bitcoin --ex_name "Comparison of models with edge propagation or without" &
# 6.6G
python main.py --model ours --seed 123 --device_id 3 --propagate inneraug --dataset wikielec --ex_name "Comparison of models with edge propagation or without" &
# 10G
python main.py --model ours --seed 123 --device_id 4 --propagate inneraug --dataset redditbody --ex_name "Comparison of models with edge propagation or without" &

python main.py --model ours --seed 222 --device_id 0 --propagate inneraug --dataset yelp --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 1 --propagate inneraug --dataset collab --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 5 --propagate inneraug --dataset bitcoin --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 6 --propagate inneraug --dataset wikielec --ex_name "Comparison of models with edge propagation or without" &
python main.py --model ours --seed 222 --device_id 7 --propagate inneraug --dataset redditbody --ex_name "Comparison of models with edge propagation or without" &
