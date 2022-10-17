#!/usr/bin/env bash
#$ -binding linear:4 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N odd-one-out
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='A100*'
#$ -l h_vmem=16G 
#$ -l mem_free=16G
#$ -t 1-35

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/ooo_heterogeneous_pretraining.sh

dataset='mnist';
out_path="/home/space/OOOPretraining/results";
data_path="/home/space/datasets/${dataset}/processed";
network='Custom';

sampling='uniform';
testing='uniform';
n_classes=10;
min_samples=3;
optim='sgd';
burnin=30;
patience=10;
steps=40;

samples=( 10 20 30 40 50 100 500 );
max_epochs=( 400 400 300 300 300 200 200 );
ooo_batch_sizes=( 32 64 64 64 64 64 128 );
main_batch_sizes=( 8 16 16 16 32 32 64 );
etas=( 0.001 0.001 0.001 0.001 0.001 0.001 0.001 );
max_triplets=( 500 1000 1000 1000 2000 2000 3000 );
seeds=( 0 1 2 3 4 );

source ~/.bashrc
conda activate triplet_ssl

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

logdir="./logs/${dataset}/${network}/${dist}/$SGE_TASK_ID";
mkdir -p $logdir;

echo "Started odd-one-out semi-supervised learning $SGE_TASK_ID for $network at $(date)"

python main.py --out_path $out_path --data_path $data_path --network $network --dataset $dataset --samples ${samples[@]} --optim $optim --sampling $sampling --min_samples $min_samples --n_classes $n_classes --max_triplets ${max_triplets[@]} --ooo_batch_sizes ${ooo_batch_sizes[@]} --main_batch_sizes ${batch_sizes[@]} --epochs ${max_epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} >> ${logdir}/${task}_${dist}.out

printf "Finished odd-one-out semi-supervised learning $SGE_TASK_ID for $network at $(date)\n"

# rm -r $cuda_dir;
