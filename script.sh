maximum_gpu_number=7
gpu=0
one=1
do_experiment(){
    python main.py --model ours --time_att_0d_dropout $d0 --time_att_2d_dropout $d2 --handling_time_att $hatt --seed $seed --device_id $gpu --propagate $propagate --dataset $dataset --ex_name "$ex_name" &
    if [ $gpu == $maximum_gpu_number ]; then gpu=0; else let gpu=$gpu+$one; fi
}
do_ex_seeds(){
    seed=117
    do_experiment
    seed=3690
    do_experiment
}

dataset=wikielec
propagate=inneraug
ex_name="Comparison of models with difference tdrops and att handling"

d0=0.1
d2=0.1
hatt="att_x_all"
do_ex_seeds
d0=0.5
d2=0.5
hatt="att_x_all"
do_ex_seeds
d0=0.0
d2=0.5
hatt="att_x_all"
do_ex_seeds
d0=0.5
d2=0.0
hatt="att_x_all"
do_ex_seeds

d0=0.1
d2=0.1
hatt="att_x_last"
do_ex_seeds
d0=0.5
d2=0.5
hatt="att_x_last"
do_ex_seeds
d0=0.0
d2=0.5
hatt="att_x_last"
do_ex_seeds
d0=0.5
d2=0.0
hatt="att_x_last"
do_ex_seeds

d0=0.1
d2=0.1
hatt="att"
do_ex_seeds
d0=0.5
d2=0.5
hatt="att"
do_ex_seeds
d0=0.0
d2=0.5
hatt="att"
do_ex_seeds
d0=0.5
d2=0.0
hatt="att"
do_ex_seeds

# kill all python processes
# lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill {}