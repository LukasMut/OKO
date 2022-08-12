#!/usr/bin/env bash
#$ -binding linear:4 # request 4 CPUs (8 with HT) (4 per GPU is recommended)
#$ -N mle_homogeneous
#$ -q all.q
#$ -cwd
#$ -l cuda=1 -l gputype='!GTX'
#$ -t 1-40

# submit as: qsub -m eas -M muttenthaler@cbs.mpg.de ./sh_scripts/mle_homogeneous.sh

ds_path='/home/space/OOOPretraining/ooo_data';
out_path='/home/space/OOOPretraining/results/CNN';
mnist_path='/home/space/datasets/mnist/processed';
network='CNN';
dist='homogeneous';
samples=( 5 10 20 30 40 50 100 );
n_classes=10;
task='mle';
bs=( 8 8 8 16 16 32 32 );
k=3;
epochs=( 50 100 200 300 400 500 1000 );
eta=0.001;
burnin=20;
ws=5;
testing='uniform';
sampling='standard';
mle_loss='standard';
steps=( 10 10 10 10 15 15 20 20 );
seeds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 );

source ~/.bashrc

cuda_dir="./cuda/infs/";
mkdir -p ${cuda_dir};

nvidia-smi > ${cuda_dir}/nvidia_smi.out
python parse_nvidia_smi.py ${cuda_dir}/nvidia_smi.out > ${cuda_dir}/cuda_version.out

input="${cuda_dir}/cuda_version.out"
while read -r line
do
	cuda_version=${line};
done < "$input"

printf "CUDA version on current GPU node: $cuda_version\n"

# conda activate ooo_pretraining_cuda_${cuda_version}

cuda_version='11.4'
conda activate ooo_pretraining_cuda_${cuda_version}

for i in "${!samples[@]}"; do
	
	logdir="./logs/${task}/${samples[$i]}/${dist}/";
	mkdir -p $logdir;

	echo "Started $task $SGE_TASK_ID for ${samples[$i]} samples at $(date)"

	python train.py --ds_path $ds_path --out_path $out_path --mnist_path $mnist_path --distribution $dist --network $network --n_samples ${samples[$i]} --n_classes $n_classes --task $task --batch_size ${bs[$i]} --k $k --epochs ${epochs[$i]} --eta $eta --burnin $burnin --window_size $ws --steps ${steps[$i]} --testing $testing --sampling $sampling --mle_loss $mle_loss --seeds ${seeds[@]} >> ${logdir}/${task}_${dist}.out

	printf "Finished $task $SGE_TASK_ID for ${samples[$i]} samples at $(date)\n"

done
