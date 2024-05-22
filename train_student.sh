#!/bin/bash

### ResNet-56: save_t_models/resnet56_cifar100_epoch_240_bs_64_lr_0.05_decay_0.0005_trial_5/resnet56_best.pth
### input units_list - dim0: 64 64 32 32 50 100 200 400

for epochs in 240
do

for model_t in resnet56
do

for model_s in resnet20
do

for dataset in cifar100
do

for bs in 64
do

for path_t in save_t_models/resnet56_cifar100_epoch_240_bs_64_lr_0.05_decay_0.0005_trial_5/resnet56_best.pth
do

for lr in 0.05
do

for m in KD
do

for kd_T in 4
do

for gamma in 1
do

for alpha in 2
do

for delta in 5
do

for trial in 1 2 3 4 5
do

for tda_layer_t in -1
do

for dim in 0
do

for operator in mean
do

    ### FC input
    python3 train_student.py --num_workers=1 --model_s ${model_s} --dataset ${dataset} --batch_size ${bs} --epochs ${epochs} --lr ${lr} --lr_decay_epochs 150 180 210 --trial ${trial} --path_t ${path_t} --distill KD --kd_T ${kd_T} --gamma ${gamma} --alpha ${alpha} -b 0.8 --TDA True --delta ${delta} --tda_layer_t ${tda_layer_t} --dim ${dim} --operator ${operator} --units_list 64 64 32 32 50 100 200 400 > ./save_s_tda_ripsnet_t_logs_final/student_result_t_${model_t}_s_${model_s}_${dataset}_${bs}_${lr}_${epochs}epochs_${m}_T_${kd_T}_g_${gamma}_a_${alpha}_d_${delta}_tda_layer_t_${tda_layer_t}_dim${dim}_${operator}_trial_${trial}.log

done

done

done

done

done

done

done

done

done

done

done

done

done

done

done

done


