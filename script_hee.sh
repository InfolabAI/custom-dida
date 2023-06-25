## 23G
#python main.py --model tokengt_cd --device_id 3 --dataset collab --ex_name "Comparison among tokengt based on CD and no CD" &
## 8G
#python main.py --model tokengt_cd --device_id 0 --dataset yelp --ex_name "Comparison among tokengt based on CD and no CD" &
## 2G
#python main.py --model tokengt_cd --device_id 0 --dataset bitcoin --ex_name "Comparison among tokengt based on CD and no CD" &
## OOM
##python main.py --model tokengt_cd --device_id 0 --dataset redditbody --ex_name "Comparison among tokengt based on CD and no CD" &
## 3G

python main.py --model tokengt_cd --device_id 0 --dataset wikielec --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cd --device_id 1 --dataset bitcoin --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cd --device_id 2 --dataset collab --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cd --device_id 3 --dataset yelp --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &

python main.py --model tokengt_cdrandom --device_id 4 --dataset wikielec --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cdrandom --device_id 5 --dataset bitcoin --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cdrandom --device_id 6 --dataset collab --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &
python main.py --model tokengt_cdrandom --device_id 7 --dataset yelp --ex_name "Comparison among tokengt_cd and tokengt_cdrandom" &