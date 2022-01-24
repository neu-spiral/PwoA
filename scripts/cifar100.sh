#!/bin/bash

dataset=cifar100
model=wrn34-10

st=irregular
preLBGAT=${dataset}_${model}_lbgat6.pt

pretrain_prune_finetune_hb_mixup() {
    xw=0
    lx=0.0000005
    ly=0.000002
    pr=$1
    premodel=$4
    save_name=prune_${pr}_${st}_teacher_LBGAT_student_LBGAT_mix_$2
    distillModel=/home/tong/MIBottleneck/MIBottleneck-Pruning/assets/models/${preLBGAT}
    run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml -m ${model} \
        -ep 50 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.01 -bs 128 \
        -reopt sgd --lr-scheduler cosine -ree 100 -relr 0.005 -relx ${lx} -rely ${ly} \
        -mf ${dataset}_${model}_${save_name}.pt \
        -lm ${premodel} \
        --distill --distill_loss kl --distill_model_path ${distillModel} --distill_temp 30 --distill_alpha 100 \
        -adv adv --mix_ratio $2 --device $3 \
        >logs/${dataset}/${model}/${save_name}.out
}


pretrain_prune_finetune_hb_mixup 0.75 0 "cuda:0" ${preLBGAT};
pretrain_prune_finetune_hb_mixup 0.75 0.2 "cuda:0" ${preLBGAT};
pretrain_prune_finetune_hb_mixup 0.75 0.4 "cuda:0" ${preLBGAT};
pretrain_prune_finetune_hb_mixup 0.75 0.6 "cuda:0" ${preLBGAT};
pretrain_prune_finetune_hb_mixup 0.75 0.8 "cuda:0" ${preLBGAT};
pretrain_prune_finetune_hb_mixup 0.75 1 "cuda:0" ${preLBGAT};
