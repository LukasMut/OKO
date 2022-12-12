# Odd-k-out learning

## Learning from sets counteracts overfitting to the mode


```bash
python main.py --out_path /path/to/results \
--data_path ./datasets/cifar10/processed \
--network Custom \
--samples 50 \
--n_classes 10 \
--k 4 \
--targets soft \
--oko_batch_sizes 64 \
--main_batch_sizes 8 \
--num_sets 3000 \
--epochs 50 \
--etas 0.001 \
--optim sgd \
--burnin 20 \
--steps 20 \
--sampling uniform \
--min_samples 5 \
--seeds 42 \
--dataset cifar10 \
--probability_masses 0.8 \
--regularization 
```
