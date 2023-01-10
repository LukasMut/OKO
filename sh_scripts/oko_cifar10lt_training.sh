#!/usr/bin/env bash
#$ -binding linear:4 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N odd-one-out
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='A100*'
#$ -l h_vmem=16G 
#$ -l mem_free=16G
#$ -t 1-120

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/ooo_heterogeneous_pretraining.sh

dataset='cifar10_lt';
out_path="/home/space/OOOPretraining/results/cifar10_lt";
# network='ResNet18';
network='Custom';

n_classes=10;
targets='hard';
optim='sgd';
burnin=35;
patience=15;
steps=40;

# sampling_strategies=( 'uniform' 'dynamic' );
sampling_strategies=( 'dynamic' );

num_odds=( 1 2 3 4 );
max_epochs=( 200 200 200 200 200 200 );
oko_batch_sizes=( 64 128 256 64 128 256 );
main_batch_sizes=( 128 128 128 128 128 128 );
etas=( 0.001 0.001 0.001 0.001 0.001 0.001 );
# etas=( 0.01 0.01 0.01 0.01 0.01 0.01 );
num_sets=( 30000 30000 30000 40000 40000 40000 );
seeds=( 0 1 2 3 4 );

source ~/.bashrc
conda activate triplet_ssl

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Started odd-k-out learning $SGE_TASK_ID for $network at $(date)"

for sampling in "${sampling_strategies[@]}"; do

	logdir="./logs/${dataset}/${network}/${sampling}/$SGE_TASK_ID";
	mkdir -p $logdir;

	python main.py --out_path $out_path --network $network --dataset $dataset --optim $optim --sampling $sampling --n_classes $n_classes --targets $targets --k ${num_odds[@]} --num_sets ${num_sets[@]} --oko_batch_sizes ${oko_batch_sizes[@]} --main_batch_sizes ${main_batch_sizes[@]} --epochs ${max_epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} --regularization >> ${logdir}/ooo_${sampling}.out

done

printf "Finished odd-k-out learning $SGE_TASK_ID for $network at $(date)\n"
