#!/usr/bin/env bash
#$ -binding linear:8 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N mle_heterogeneous_batch_balancing
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='A100G40'
#$ -l h=node45
#$ -t 1-110

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/mle_heterogeneous.sh

network='VGG16';
dataset='cifar10';
# out_path="/home/space/OOOPretraining/results/${dataset}/batch_balancing_label_smoothing";
out_path="/home/space/OOOPretraining/results/${dataset}/batch_balancing";
data_path="/home/space/datasets/${dataset}/processed";
dist='heterogeneous';
task='mle';
sampling='uniform';
testing='uniform';
k=3;
min_samples=3;

# shapes=( 0 1 2 3 4 5 );
shape=0;
n_classes=10;
samples=( 5 10 20 30 40 50 100 500 1000 2000 4000 );
bs=( 8 8 8 16 16 32 32 64 128 128 256 );
epochs=( 400 400 400 300 300 300 200 200 200 100 100 );

# eta=0.0001;
# optim='adam';
eta=0.001;
optim='sgd';

mle_loss='standard';
steps=100;
seeds=( 0 1 2 3 4 5 6 7 8 9 );

source ~/.bashrc
conda activate ooo_pretraining_cuda

# export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75

logdir="./logs/${dataset}/${task}/${network}/${dist}/vanilla/${SGE_TASK_ID}";
mkdir -p $logdir;

echo "Started $task $SGE_TASK_ID for $dataset with $network at $(date)"

python main.py --out_path $out_path --data_path $data_path --network $network --dataset $dataset --distribution $dist --sampling $sampling --shape $shape --optim $optim --mle_loss $mle_loss --samples ${samples[@]} --batch_size ${bs[@]} --n_classes $n_classes --sampling $sampling --testing $testing --task $task --k $k --epochs ${epochs[@]} --eta $eta --min_samples $min_samples --steps $steps --seeds ${seeds[@]} >> ${logdir}/${task}_${dist}.out

printf "Finished $task $SGE_TASK_ID for $dataset with $network at $(date)\n"

