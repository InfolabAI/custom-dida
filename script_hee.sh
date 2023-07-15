
python main.py --model ours --seed 123 --device_id 0 --hidden_augment dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --hidden_augment dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --hidden_augment dyaug --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --hidden_augment dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --hidden_augment dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --hidden_augment dyaug --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --hidden_augment dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --hidden_augment dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --hidden_augment dyaug --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &


# normal
python main.py --model ours --seed 123 --device_id 0 --hidden_augment no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --hidden_augment no --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --hidden_augment no --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --hidden_augment no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --hidden_augment no --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --hidden_augment no --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --hidden_augment no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --hidden_augment no --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --hidden_augment no --dataset collab --ex_name "Dynamic aug" &

exit
# normal
python main.py --model ours --seed 123 --device_id 0 --hidden_augment dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --hidden_augment dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --hidden_augment dyaug --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --hidden_augment dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --hidden_augment dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --hidden_augment dyaug --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --hidden_augment dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --hidden_augment dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --hidden_augment dyaug --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
