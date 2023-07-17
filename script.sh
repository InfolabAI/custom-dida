
python main.py --model ours --seed 123 --device_id 0 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &


# normal
python main.py --model ours --seed 123 --device_id 0 --propagate no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --propagate no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate no --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate no --dataset collab --ex_name "Dynamic aug" &

# normal
python main.py --model ours --seed 123 --device_id 0 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
