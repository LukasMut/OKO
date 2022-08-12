#!/usr/bin/env bash
#$ -binding linear:8 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N ooo_heterogeneous_pretraining
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='A100*'
#S -l h=node53
#$ -l h_vmem=16G 
#$ -l mem_free=16G
#$ -t 1-240

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/ooo_heterogeneous_pretraining.sh

dataset='mnist';
out_path="/home/space/OOOPretraining/results/${dataset}";
data_path="/home/space/datasets/${dataset}/processed";
network='Custom';
task='ooo_dist';
dist='heterogeneous';

sampling='uniform';
testing='uniform';
n_classes=3;
min_samples=3;
max_triplets=150000;
optim='sgd';
burnin=50;
patience=10;
steps=40;

shapes=( 0 1 2 3 4 5 );
samples=( 5 10 20 30 40 50 100 500 );
epochs=( 200 200 200 200 200 200 200 200 );
batch_sizes=( 256 256 256 256 256 256 256 256 );
etas=( 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 );
seeds=( 0 1 2 3 4 );

source ~/.bashrc
conda activate triplet_ssl

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform


logdir="./logs/${task}_pretraining/${dataset}/${network}/${dist}/$SGE_TASK_ID";
mkdir -p $logdir;
	
echo "Started $task pretraining $SGE_TASK_ID for $network at $(date)"

python main.py --out_path $out_path --data_path $data_path --network $network --dataset $dataset --task $task --distribution $dist --shapes ${shapes[@]} --samples ${samples[@]} --optim $optim --sampling $sampling --min_samples $min_samples --max_triplets $max_triplets --n_classes $n_classes --batch_sizes ${batch_sizes[@]} --epochs ${epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} >> ${logdir}/${task}_${dist}_pretraining.out

printf "Finished $task pretraining $SGE_TASK_ID for $network at $(date)\n"
