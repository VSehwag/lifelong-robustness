dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

eval_base() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4)
    CUDA_VISIBLE_DEVICES=$3 python -u eval.py --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2 --evaluator adv --batch-size 64 --print-freq 10 --ckpt ./trained_models/imagenette2_$2_trainadv_$1/trial_0/checkpoint/checkpoint.pth.tar --eval-attack $4  --exp-name imagenette2_eval_$2_finetune_pretrainadv_$1_finetuneadv_$4
}

eval_finetune() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4)
    CUDA_VISIBLE_DEVICES=$3 python -u eval.py --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2 --evaluator adv --batch-size 64 --print-freq 10 --ckpt ./trained_models/imagenette2_$2_finetune_pretrainadv_$1_finetuneadv_$4/trial_0/checkpoint/checkpoint.pth.tar --eval-attack $4  --exp-name imagenette2_eval_$2_finetune_pretrainadv_$1_finetuneadv_$4
}


# eval_finetune none ResNet18 1 linf &
# eval_finetune none ResNet18 2 l2 &
# eval_finetune none ResNet18 3 jpeg &
# eval_finetune none ResNet18 6 gabor &
# eval_finetune none ResNet18 7 snow ;

# eval_base linf ResNet18 2 linf &
# eval_finetune linf ResNet18 3 l2 &
# eval_finetune linf ResNet18 4 jpeg &
# eval_finetune linf ResNet18 5 gabor &
# eval_finetune linf ResNet18 6 snow ;
# wait;

# eval_finetune l2 ResNet18 2 linf &
# eval_base l2 ResNet18 3 l2 &
# eval_finetune l2 ResNet18 4 jpeg &
# eval_finetune l2 ResNet18 5 gabor &
# eval_finetune l2 ResNet18 6 snow ;
# wait;

# eval_finetune jpeg ResNet18 2 linf &
# eval_finetune jpeg ResNet18 3 l2 &
# eval_base jpeg ResNet18 4 jpeg &
# eval_finetune jpeg ResNet18 5 gabor &
# eval_finetune jpeg ResNet18 6 snow ;
# wait;

# eval_finetune gabor ResNet18 2 linf &
# eval_finetune gabor ResNet18 3 l2 &
# eval_finetune gabor ResNet18 4 jpeg &
# eval_base gabor ResNet18 5 gabor &
# eval_finetune gabor ResNet18 6 snow ;
# wait;

# eval_finetune snow ResNet18 2 linf &
# eval_finetune snow ResNet18 3 l2 &
# eval_finetune snow ResNet18 4 jpeg &
# eval_finetune snow ResNet18 5 gabor &
# eval_base snow ResNet18 6 snow &
# wait; 



eval_base_pretrain_only() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4)
    CUDA_VISIBLE_DEVICES=$3 python -u eval.py --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2 --evaluator adv --batch-size 64 --print-freq 10 --ckpt ./trained_models/imagenette2_$2_trainadv_$1/trial_0/checkpoint/checkpoint.pth.tar --eval-attack $1  --exp-name imagenette2_eval_$2_finetune_pretrainadv_$1_finetuneadv_$4_pretrain_only
}

eval_finetune_pretrain_only() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4)
    CUDA_VISIBLE_DEVICES=$3 python -u eval.py --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2 --evaluator adv --batch-size 64 --print-freq 10 --ckpt ./trained_models/imagenette2_$2_finetune_pretrainadv_$1_finetuneadv_$4/trial_0/checkpoint/checkpoint.pth.tar --eval-attack $1  --exp-name imagenette2_eval_$2_finetune_pretrainadv_$1_finetuneadv_$4_pretrain_only
}


eval_finetune_pretrain_only none ResNet18 1 linf &
eval_finetune_pretrain_only none ResNet18 2 l2 &
eval_finetune_pretrain_only none ResNet18 3 jpeg &
eval_finetune_pretrain_only none ResNet18 6 gabor &
eval_finetune_pretrain_only none ResNet18 7 snow ;

eval_base_pretrain_only linf ResNet18 2 linf &
eval_finetune_pretrain_only linf ResNet18 3 l2 &
eval_finetune_pretrain_only linf ResNet18 4 jpeg &
eval_finetune_pretrain_only linf ResNet18 5 gabor &
eval_finetune_pretrain_only linf ResNet18 6 snow ;
wait;

eval_finetune_pretrain_only l2 ResNet18 2 linf &
eval_base_pretrain_only l2 ResNet18 3 l2 &
eval_finetune_pretrain_only l2 ResNet18 4 jpeg &
eval_finetune_pretrain_only l2 ResNet18 5 gabor &
eval_finetune_pretrain_only l2 ResNet18 6 snow ;
wait;

eval_finetune_pretrain_only jpeg ResNet18 2 linf &
eval_finetune_pretrain_only jpeg ResNet18 3 l2 &
eval_base_pretrain_only jpeg ResNet18 4 jpeg &
eval_finetune_pretrain_only jpeg ResNet18 5 gabor &
eval_finetune_pretrain_only jpeg ResNet18 6 snow ;
wait;

eval_finetune_pretrain_only gabor ResNet18 2 linf &
eval_finetune_pretrain_only gabor ResNet18 3 l2 &
eval_finetune_pretrain_only gabor ResNet18 4 jpeg &
eval_base_pretrain_only gabor ResNet18 5 gabor &
eval_finetune_pretrain_only gabor ResNet18 6 snow ;
wait;

eval_finetune_pretrain_only snow ResNet18 2 linf &
eval_finetune_pretrain_only snow ResNet18 3 l2 &
eval_finetune_pretrain_only snow ResNet18 4 jpeg &
eval_finetune_pretrain_only snow ResNet18 5 gabor &
eval_base_pretrain_only snow ResNet18 6 snow &
wait; 