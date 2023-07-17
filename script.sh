# our method with edge propagation with alpha_std
python main.py --model ours --seed 123 --device_id 0 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
exit

python main.py --model ours --seed 222 --device_id 1 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &

# our method with edge propagation without alpha_std
python main.py --model ours --seed 123 --device_id 0 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &


# our method without dynamic augmentation
python main.py --model ours --seed 123 --device_id 0 --propagate no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --propagate no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --propagate no --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --propagate no --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --propagate no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --propagate no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --propagate no --dataset collab --ex_name "Dynamic aug" &

# DIDA
python main.py --model dida --seed 123 --device_id 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model dida --seed 222 --device_id 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model dida --seed 321 --device_id 2 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model dida --seed 123 --device_id 3 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model dida --seed 222 --device_id 4 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model dida --seed 321 --device_id 5 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model dida --seed 123 --device_id 6 --dataset collab --ex_name "Dynamic aug" &
python main.py --model dida --seed 222 --device_id 7 --dataset collab --ex_name "Dynamic aug" &
python main.py --model dida --seed 321 --device_id 0 --dataset collab --ex_name "Dynamic aug" &