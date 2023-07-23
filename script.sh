# 3.1G
python main.py --model ours --seed 123 --device_id 1 --propagate no --dataset yelp --ex_name "Evaluation analysis" &
# 3.1G
python main.py --model ours --seed 123 --device_id 2 --propagate no --dataset collab --ex_name "Evaluation analysis" &
# 4.5G
python main.py --model ours --seed 123 --device_id 3 --propagate no --dataset bitcoin --ex_name "Evaluation analysis" &
# 6.6G
python main.py --model ours --seed 123 --device_id 4 --propagate no --dataset wikielec --ex_name "Evaluation analysis" &
# 10G
python main.py --model ours --seed 123 --device_id 5 --propagate no --dataset redditbody --ex_name "Evaluation analysis" &
