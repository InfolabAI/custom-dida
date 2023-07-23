# 3.1G
python main.py --model ours --seed 123 --device_id 0 --propagate no --dataset yelp --ex_name "10.Comparison between normal and inneraug" &
# 3.1G
python main.py --model ours --seed 123 --device_id 1 --propagate no --dataset collab --ex_name "10.Comparison between normal and inneraug" &
# 4.3G
python main.py --model ours --seed 123 --device_id 2 --propagate no --dataset bitcoin --ex_name "10.Comparison between normal and inneraug" &
# 6G
python main.py --model ours --seed 123 --device_id 3 --propagate no --dataset wikielec --ex_name "10.Comparison between normal and inneraug" &
# 12.6G
python main.py --model ours --seed 123 --device_id 4 --propagate no --dataset redditbody --ex_name "10.Comparison between normal and inneraug" &

# 3.1G
python main.py --model ours --seed 123 --device_id 0 --propagate inneraug --dataset yelp --ex_name "10.Comparison between normal and inneraug" &
# 3.1G
python main.py --model ours --seed 123 --device_id 1 --propagate inneraug --dataset collab --ex_name "10.Comparison between normal and inneraug" &
# 4.3G
python main.py --model ours --seed 123 --device_id 2 --propagate inneraug --dataset bitcoin --ex_name "10.Comparison between normal and inneraug" &
# 6G
python main.py --model ours --seed 123 --device_id 3 --propagate inneraug --dataset wikielec --ex_name "10.Comparison between normal and inneraug" &
# 12.6G
python main.py --model ours --seed 123 --device_id 5 --propagate inneraug --dataset redditbody --ex_name "10.Comparison between normal and inneraug" &