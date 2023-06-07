#python main.py --model tokengt --device_id 7 --dataset collab --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 6 --dataset yelp --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 5 --dataset collab --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 4 --dataset yelp --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &

#python main.py --model tokengt --device_id 7 --dataset bitcoin --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 6 --dataset redditbody --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 5 --dataset wikielec --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 4 --dataset bitcoin --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 2 --dataset wikielec --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &

python main.py --model tokengt_nocd --device_id 7 --dataset collab --ex_name "Comparison among tokengt based on CD and no CD" &
python main.py --model tokengt_nocd --device_id 6 --dataset yelp --ex_name "Comparison among tokengt based on CD and no CD" &
python main.py --model tokengt_nocd --device_id 5 --dataset bitcoin --ex_name "Comparison among tokengt based on CD and no CD" &
python main.py --model tokengt_nocd --device_id 5 --dataset redditbody --ex_name "Comparison among tokengt based on CD and no CD" &
python main.py --model tokengt_nocd --device_id 4 --dataset wikielec --ex_name "Comparison among tokengt based on CD and no CD" &

# 23G
python main.py --model tokengt_cd --device_id 3 --dataset collab --ex_name "Comparison among tokengt based on CD and no CD" &
# 8G
python main.py --model tokengt_cd --device_id 0 --dataset yelp --ex_name "Comparison among tokengt based on CD and no CD" &
# 2G
python main.py --model tokengt_cd --device_id 0 --dataset bitcoin --ex_name "Comparison among tokengt based on CD and no CD" &
# OOM
#python main.py --model tokengt_cd --device_id 0 --dataset redditbody --ex_name "Comparison among tokengt based on CD and no CD" &
# 3G
python main.py --model tokengt_cd --device_id 0 --dataset wikielec --ex_name "Comparison among tokengt based on CD and no CD" &

python main.py --model tokengt_cd --device_id 2 --shuffled 1 --dataset collab --ex_name "Comparison among tokengt based on CD and no CD (shuffled)" &
python main.py --model tokengt_cd --device_id 1 --shuffled 1 --dataset yelp --ex_name "Comparison among tokengt based on CD and no CD (shuffled)" &
python main.py --model tokengt_cd --device_id 1 --shuffled 1 --dataset bitcoin --ex_name "Comparison among tokengt based on CD and no CD (shuffled)" &
#python main.py --model tokengt_cd --device_id 1 --shuffled 1 --dataset redditbody --ex_name "Comparison among tokengt based on CD and no CD (shuffled)" &
python main.py --model tokengt_cd --device_id 1 --shuffled 1 --dataset wikielec --ex_name "Comparison among tokengt based on CD and no CD (shuffled)" &

#python main.py --model dida --device_id 4 --dataset collab --ex_name "Comparison based on the shuffle option" &
#python main.py --model dida --device_id 2 --dataset yelp --ex_name "Comparison based on the shuffle option" &
#python main.py --model dida --device_id 4 --dataset bitcoin --ex_name "Comparison based on the shuffle option" &
#python main.py --model dida --device_id 2 --dataset wikielec --ex_name "Comparison based on the shuffle option" &


#python main.py --dataset wikielec --draw_community_detection 1 &
#python main.py --dataset bitcoin --draw_community_detection 1 &
#python main.py --dataset redditbody --draw_community_detection 1 &
#python main.py --dataset collab --draw_community_detection 1 &
#python main.py --dataset yelp --draw_community_detection 1 &