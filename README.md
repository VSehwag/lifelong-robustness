## Lifelong robustness

A minimialistic imlementation of adversarial training on MNIST/FMNIST/CIFAR10 dataset. Defaults are set to CIFAR10 dataset.

Currently support only l2 and linf attacks. Future target is to add l1 and a spatial attack. 

1. Training a network
`CUDA_VISIBLE_DEVICES=0,1 python train.py --trainer adv --train-attack linf --evaluator adv --eval-attack linf --exp-name cifar10_cnnLarge_adv_linf`

2. Evaluating a network
`CUDA_VISIBLE_DEVICES=0 python eval.py --evaluator adv --eval-attack linf --ckpt checkpoint_path`
You can pass `--autoattack` flag to evaluate with autoattack.