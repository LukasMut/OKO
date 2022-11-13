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

dataset='cifar10';
out_path="/home/space/OOOPretraining/results";
data_path="/home/space/datasets/${dataset}/processed";
network='Custom';

testing='uniform';
n_classes=10;
min_samples=5;
optim='sgd';
burnin=35;
patience=15;
steps=40;

sampling_strategies=( 'uniform' 'dynamic' );
probability_masses=( 0.8 0.85 0.9 0.95 ); 
samples=( 40 50 100 500 1000 2000 );
max_epochs=( 300 300 300 200 200 200 );
ooo_batch_sizes=( 128 128 128 128 128 256 );
main_batch_sizes=( 16 16 32 32 64 128 );
etas=( 0.001 0.001 0.001 0.001 0.001 0.001 );
max_triplets=( 2000 3000 4000 5000 10000 20000 );
seeds=( 0 1 2 3 4 );

source ~/.bashrc
conda activate triplet_ssl

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Started odd-one-out learning $SGE_TASK_ID for $network at $(date)"

for sampling in "${sampling_strategies[@]}"; do

	logdir="./logs/${dataset}/${network}/${sampling}/$SGE_TASK_ID";
	mkdir -p $logdir;

	python main.py --out_path $out_path --data_path $data_path --network $network --dataset $dataset --samples ${samples[@]} --optim $optim --sampling $sampling --min_samples $min_samples --probability_masses ${probability_masses[@]} --n_classes $n_classes --max_triplets ${max_triplets[@]} --ooo_batch_sizes ${ooo_batch_sizes[@]} --main_batch_sizes ${main_batch_sizes[@]} --epochs ${max_epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} --regularization >> ${logdir}/ooo_${sampling}.out

done

printf "Finished odd-one-out learning $SGE_TASK_ID for $network at $(date)\n"
