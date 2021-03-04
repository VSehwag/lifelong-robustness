dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

# no warmup, only 50 epochs, using resnet18 network

training_loop_base() {
    # train-attack ($1), arch ($2), gpu ($3)
    CUDA_VISIBLE_DEVICES=$3 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --trainer base --evaluator base --print-freq 10 --arch $2 --batch-size 128 --exp-name imagenette2_$2_base --trial 0 --epochs 50 --warmup | tee -a ./logs/imagenette2_$2_base.txt
}

training_loop() {
    # train-attack ($1), arch ($2), gpu ($3)
    CUDA_VISIBLE_DEVICES=$3 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2  --trainer adv  --train-attack $1 --evaluator base --batch-size 128 --exp-name imagenette2_$2_trainadv_$1 --trial 0 --epochs 50 --print-freq 10 --warmup | tee -a ./logs/imagenette2_$2_trainadv_$1.txt
}


finetuning_loop() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4), trial ($5), lr ($6)
    CUDA_VISIBLE_DEVICES=$3 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2  --trainer adv  --train-attack $4 --evaluator base --batch-size 128 --exp-name imagenette2_$2_finetune_pretrainadv_$1_finetuneadv_$4_lr_$5 --epochs 50 --print-freq 10 --lr $5 --ckpt ./trained_models/imagenette2_$2_trainadv_$1/trial_0/checkpoint/checkpoint.pth.tar --trial $5 | tee -a ./logs/imagenette2_$2_trainadv_$1.txt

}

finetuning_loop_wamrup() {
    # pretrain-attack ($1), arch ($2), gpu ($3), finetune-attack ($4), trial ($5)
    CUDA_VISIBLE_DEVICES=$3 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --arch $2  --trainer adv  --train-attack $4 --evaluator base --batch-size 128 --exp-name imagenette2_$2_finetune_pretrainadv_$1_finetuneadv_$4_warmup --epochs 50 --print-freq 10 --lr 0.01 --ckpt ./trained_models/imagenette2_$2_trainadv_$1/trial_0/checkpoint/checkpoint.pth.tar --warmup --trial $5 | tee -a ./logs/imagenette2_$2_trainadv_$1.txt

}

# training_loop_base none ResNet18 0 &
# training_loop linf ResNet18 1 &
# training_loop l2 ResNet18 2 &
#training_loop jpeg ResNet18 3 &
#training_loop gabor ResNet18 4 &
#training_loop snow ResNet18 5 ;
#wait;

#finetuning_loop none ResNet18 2 linf &
#finetuning_loop none ResNet18 3 l2 &
#finetuning_loop none ResNet18 4 jpeg &
#finetuning_loop none ResNet18 5 gabor &
#finetuning_loop none ResNet18 6 snow ;
#wait;


## New set of experiments
finetuning_loop linf ResNet18 3 l2 0 0.001 &
finetuning_loop linf ResNet18 4 jpeg 0 0.001 &
finetuning_loop linf ResNet18 5 gabor 0 0.001 &
finetuning_loop linf ResNet18 6 snow 0 0.001 ;
wait;

finetuning_loop l2 ResNet18 3 linf 0 0.001 &
finetuning_loop l2 ResNet18 4 jpeg 0 0.001 &
finetuning_loop l2 ResNet18 5 gabor 0 0.001 &
finetuning_loop l2 ResNet18 6 snow 0 0.001 ;
wait;

finetuning_loop jpeg ResNet18 3 linf 0 0.001 &
finetuning_loop jpeg ResNet18 4 l2 0 0.001 &
finetuning_loop jpeg ResNet18 5 gabor 0 0.001 &
finetuning_loop jpeg ResNet18 6 snow 0 0.001 ;
wait;

finetuning_loop gabor ResNet18 3 linf 0 0.001 &
finetuning_loop gabor ResNet18 4 l2 0 0.001 &
finetuning_loop gabor ResNet18 5 jpeg 0 0.001 &
finetuning_loop gabor ResNet18 6 snow 0 0.001 ;
wait; 

finetuning_loop snow ResNet18 3 linf 0 0.001 &
finetuning_loop snow ResNet18 4 l2 0 0.001 &
finetuning_loop snow ResNet18 5 jpeg 0 0.001 &
finetuning_loop snow ResNet18 6 gabor 0 0.001 ;
wait; 

# ## debugging
# finetuning_loop linf ResNet18 4 gabor 1 &
# finetuning_loop jpeg ResNet18 5 gabor 1 &
# finetuning_loop_wamrup linf ResNet18 6 gabor 0 &
# finetuning_loop_wamrup jpeg ResNet18 7 gabor 0 ;