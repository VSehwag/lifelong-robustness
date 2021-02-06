## Lifelong robustness

A minimialistic imlementation of adversarial training on MNIST/FMNIST/CIFAR10/ImageNet dataset. Defaults are set to CIFAR10 dataset.

Currently support l2, linf, snow, gabor, and jpeg (l2) attacks.

### Thing to know, before you start editing
1. To simply the pipeline, we assume that each image have the [0., 1.] pixel ranges. If it needs normalization or scaling up to [0, 255], please do so in the the model/attacks itself.



### How to use it
1. Training a network
`CUDA_VISIBLE_DEVICES=0,1 python train.py --trainer adv --train-attack linf --evaluator adv --eval-attack linf --exp-name cifar10_cnnLarge_adv_linf`

2. Evaluating a network
`CUDA_VISIBLE_DEVICES=0 python eval.py --evaluator adv --eval-attack linf --ckpt checkpoint_path`
You can pass `--autoattack` flag to evaluate with autoattack.



### Misc
1. We use "ResNet, MobileNet" naming convention for ImageNet scale models while "resnet, mobilenet" for it cifar10 scale counterparts.
2. Eval attack steps for most attacks are set to very high values in the configs. Make sure not to eval with them after every epoch, i.e., better to set `--evaluator base` for these attacks and conduct an evaluation post training. 
3. 
