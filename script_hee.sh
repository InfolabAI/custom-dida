#python main.py --model tokengt --device_id 7 --dataset collab --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 6 --dataset yelp --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 5 --dataset collab --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 4 --dataset yelp --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &

#python main.py --model tokengt --device_id 7 --dataset bitcoin --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 6 --dataset redditbody --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model tokengt --device_id 5 --dataset wikielec --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 4 --dataset bitcoin --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &
#python main.py --model dida --device_id 2 --dataset wikielec --plot_hub_nodes 1 --ex_name "Plot based on hub nodes" &

python main.py --model tokengt --device_id 3 --dataset bitcoin --shuffled 1 --ex_name "Comparison based on the shuffle option" &
python main.py --model tokengt --device_id 1 --dataset redditbody --shuffled 1 --ex_name "Comparison based on the shuffle option" &
python main.py --model tokengt --device_id 2 --dataset wikielec --shuffled 1 --ex_name "Comparison based on the shuffle option" &
python main.py --model dida --device_id 4 --dataset bitcoin --shuffled 1 --ex_name "Comparison based on the shuffle option" &
python main.py --model dida --device_id 2 --dataset wikielec --shuffled 1 --ex_name "Comparison based on the shuffle option" &