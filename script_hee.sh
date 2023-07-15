# normal
python main.py --model ours --seed 123 --device_id 0 --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --alpha_std 1 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --alpha_std 1 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --alpha_std 1 --dataset collab --ex_name "Dynamic aug" &

exit

python main.py --model ours --seed 123 --device_id 0 --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 1 --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 2 --alpha_std 0 --dataset yelp --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 3 --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 4 --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 5 --alpha_std 0 --dataset bitcoin --ex_name "Dynamic aug" &

python main.py --model ours --seed 123 --device_id 6 --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 222 --device_id 7 --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &
python main.py --model ours --seed 321 --device_id 0 --alpha_std 0 --dataset collab --ex_name "Dynamic aug" &

