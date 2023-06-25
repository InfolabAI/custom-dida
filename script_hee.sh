## 23G
#python main.py --model tokengt_cd --device_id 3 --dataset collab --ex_name "Comparison among tokengt based on CD and no CD" &
## 8G
#python main.py --model tokengt_cd --device_id 0 --dataset yelp --ex_name "Comparison among tokengt based on CD and no CD" &
## 2G
#python main.py --model tokengt_cd --device_id 0 --dataset bitcoin --ex_name "Comparison among tokengt based on CD and no CD" &
## OOM
##python main.py --model tokengt_cd --device_id 0 --dataset redditbody --ex_name "Comparison among tokengt based on CD and no CD" &
## 3G

python main.py --model tokengt_cd --dataset wikielec --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cd --dataset bitcoin --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cd --dataset redditbody --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cd --dataset collab --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cd --dataset yelp --plot_sparsity_mat_cd 1 &

python main.py --model tokengt_cdrandom --dataset wikielec --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cdrandom --dataset bitcoin --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cdrandom --dataset redditbody --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cdrandom --dataset collab --plot_sparsity_mat_cd 1 &
python main.py --model tokengt_cdrandom --dataset yelp --plot_sparsity_mat_cd 1 &