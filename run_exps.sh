dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

# no warmup, only 50 epochs, using resnet18 network

training_loop_base() {
    # train-attack ($1), arch ($2)
    CUDA_VISIBLE_DEVICES=2,3,4,5 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --trainer base --evaluator base --print-freq 10 --arch ResNet18 --batch-size 128 --exp-name imagenette2_$2_base --trial 0 --epochs 50 | tee -a ./logs/imagenette2_$2_base.txt
mk
}

training_loop() {
    # train-attack ($1), arch ($2)
    CUDA_VISIBLE_DEVICES=2,3,4,5 python -u train.py  --configs configs/configs_imagenet.yml --dataset imagenet --datadir /data/nvme/vvikash/datasets/imagenet_subsets/imagenette2/  --trainer adv --print-freq 10 --train-attack linf --arch ResNet18 --evaluator base --batch-size 128 --exp-name imagenette2_$2_trainadv_$1 --trial 0 --epochs 50 | tee -a ./logs/imagenette2_$2_trainadv_$1.txt
mk
}
training_loop_base none ResNet18 &
training_loop linf ResNet18 &
training_loop l2 ResNet18 &
training_loop jpeg ResNet18 &
training_loop gabor ResNet18 ;
wait;
training_loop snow ResNet18 ;