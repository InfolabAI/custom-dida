python main.py --model tokengt_cd --seed 123 --augment edgeprop --device_id 0 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --augment edgeprop --device_id 0 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --augment edgeprop --device_id 6 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &

python main.py --model tokengt_cd --seed 123 --augment edgeprop --device_id 1 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --augment edgeprop --device_id 1 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --augment edgeprop --device_id 1 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &

python main.py --model tokengt_cd --seed 123 --augment edgeprop --device_id 2 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --augment edgeprop --device_id 2 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --augment edgeprop --device_id 7 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &

python main.py --model tokengt_cd --seed 123 --hidden_augment pool --device_id 3 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --hidden_augment pool --device_id 3 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --hidden_augment pool --device_id 6 --dataset yelp --ex_name "Augmentation_test_edgeprop_poolingatt" &

python main.py --model tokengt_cd --seed 123 --hidden_augment pool --device_id 4 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --hidden_augment pool --device_id 4 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --hidden_augment pool --device_id 4 --dataset bitcoin --ex_name "Augmentation_test_edgeprop_poolingatt" &

python main.py --model tokengt_cd --seed 123 --hidden_augment pool --device_id 5 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 222 --hidden_augment pool --device_id 5 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &
python main.py --model tokengt_cd --seed 321 --hidden_augment pool --device_id 7 --dataset collab --ex_name "Augmentation_test_edgeprop_poolingatt" &