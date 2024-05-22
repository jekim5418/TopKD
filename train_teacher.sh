#!/bin/bash

### ResNet
for trial in 1 2 3 4 5
do

for epochs in 240
do

for bs in 64
do

for lr in 0.05
do

for model in resnet56
do

for dataset in cifar100
do

    python3 train_teacher.py --num_workers=1 --epochs ${epochs} --batch_size ${bs} --lr ${lr} --lr_decay_epochs 150 180 210 --model ${model} --dataset ${dataset} --trial ${trial} > ./save_t_logs/teacher_result_model_${model}_dataset_${dataset}_bs_${bs}_lr_${lr}_${epochs}epochs_trial_${trial}.log

done

done

done

done

done

done
