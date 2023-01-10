#!/usr/bin/env bash
#$ -binding linear:8 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N odd-k-out
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='A100*'
#$ -l h_vmem=16G 
#$ -l mem_free=16G
#$ -t 1-540

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/ooo_heterogeneous_pretraining.sh

dataset='cifar10';
out_path="/home/space/OOOPretraining/results/${dataset}";

network='ResNet34';
# network='Custom';

n_classes=10;
targets='hard';
min_samples=5;
optim='sgd';
burnin=30;
patience=15;
steps=40;

num_odds=( 1 2 3 );
sampling_policies=( 'uniform' 'dynamic' );
probability_masses=( 0.8 0.85 0.9 );
samples=( 40 50 100 500 1000 2000 );
max_epochs=( 100 100 100 200 200 200 );
oko_batch_sizes=( 128 128 128 256 256 256 );
main_batch_sizes=( 128 128 128 256 256 512 );

# etas=( 0.001 0.001 0.001 0.001 0.001 0.001 );
# etas=( 0.01 0.01 0.01 0.01 0.01 0.01 );
etas=( 0.1 0.1 0.1 0.1 0.1 0.1 );

num_sets=( 2000 3000 4000 5000 10000 20000 );
seeds=( 0 1 2 3 4 );

source ~/.bashrc
conda activate triplet_ssl

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Started odd-k-out learning $SGE_TASK_ID for $network at $(date)"

logdir="./logs/${dataset}/${network}/${targets}/${SGE_TASK_ID}";

mkdir -p $logdir;

python main.py --out_path $out_path --network $network --dataset $dataset --samples ${samples[@]} --optim $optim --sampling ${sampling_policies[@]} --min_samples $min_samples --probability_masses ${probability_masses[@]} --n_classes $n_classes --targets $targets --k ${num_odds[@]} --num_sets ${num_sets[@]} --oko_batch_sizes ${oko_batch_sizes[@]} --main_batch_sizes ${main_batch_sizes[@]} --epochs ${max_epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} --regularization --apply_augmentations >> ${logdir}/oko_${dataset}.out

printf "Finished odd-k-out learning $SGE_TASK_ID for $network at $(date)\n"
