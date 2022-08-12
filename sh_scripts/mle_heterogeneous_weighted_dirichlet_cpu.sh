#!/usr/bin/env bash
#$ -binding linear:8 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N mle_heterogeneous_dirichlet
#$ -l h_vmem=128G
#$ -l mem_free=128G
#$ -q all.q
#$ -cwd
#$ -t 1-240

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/mle_heterogeneous.sh

ds_path='/home/space/OOOPretraining/ooo_data';
out_path='/home/space/OOOPretraining/results/dirichlet';
mnist_path='/home/space/datasets/mnist/processed';
dist='heterogeneous';
network='CNN';
shapes=( 0 1 2 3 4 5 );
samples=( 5 10 20 30 40 50 100 );
sampling='standard';
n_classes=10;
task='mle';
k=3;
epochs=( 100 100 200 300 400 500 1000 );
#bs=( 4 8 16 16 32 32 64 );
bs=( 8 8 8 16 16 32 32 );
eta=0.001;
burnin=20;
min_samples=3;
weighting='dirichlet';
ws=5;
steps=( 5 10 10 10 15 15 20 20 );
seeds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 );
# scalings=( '1.0' '5.0' '10.0' '50.0' );
# scalings=( '0.5' '100.0' );
scalings=( '50.0' );

source ~/.bashrc
conda activate ooo_pretraining_cpu

for c in "${scalings[@]}"; do	
	
	current_out="${out_path}_${c}";

	for i in "${!samples[@]}"; do
	
		logdir="./logs/${task}/${samples[$i]}/${dist}/";
		mkdir -p $logdir;
	
		echo "Started $task $SGE_TASK_ID for ${samples[$i]} samples at $(date)"

		python train.py --ds_path $ds_path --out_path $current_out --mnist_path $mnist_path --distribution $dist --network $network --shapes ${shapes[@]} --n_samples ${samples[$i]} --n_classes $n_classes --sampling $sampling --task $task --batch_size ${bs[$i]} --k $k --epochs ${epochs[$i]} --eta $eta --burnin $burnin --min_samples $min_samples --mle_loss $weighting --c $c --window_size $ws --steps ${steps[$i]} --seeds ${seeds[@]} >> ${logdir}/${task}_${dist}.out

		printf "Finished $task $SGE_TASK_ID for ${samples[$i]} samples at $(date)\n"
	done
done
