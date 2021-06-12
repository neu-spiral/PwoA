#!/bin/bash
gpu=$1
export CUDA_VISIBLE_DEVICES=${gpu}

dataset=cifar10
model=resnet18

st=irregular
preCe=${dataset}_${model}_xw_1_lx_0_ly_0_adv_best.pt
preHBaR=${dataset}_${model}_xw_1_lx_0.0005_ly_0.005_adv_best.pt
preTRADES=${dataset}_${model}_trades_hsic_lx_0_lx_0.0005_ly_0.005.pt
distillModel=${preHBaR}

xw=0
lx=(0.005)
ly=0.05

pr=(0.5 0.75 0.875 0.9375)

for ((i=0;i<${#pr[@]};i++)) do
    for ((j=0;j<${#lx[@]};j++)) do
        save_name=prune_${pr[i]}_${st}_xw_${xw}_lx_${lx[j]}_ly_${ly}_hb_distillSgd_hbSgd
        run_hsicbt -cfg config/general-prune-hsicbt-${dataset}-${model}.yaml -slmo -pr ${pr[i]} -st $st -xw $xw \
            -op sgd --lr-scheduler default -lx ${lx[j]} -ly ${ly} \
            -reopt sgd -ree 100 -relr 0.01 -relx ${lx[j]} -rely ${ly} \
            -mf ${dataset}_${model}_${save_name}.pt \
            -lm $preHBaR \
            --distill --distill_loss kl --distill_model_path ${distillModel} --distill_temp 30 --distill_alpha 1 \
            >logs/${dataset}/${model}/${save_name}.out
    done
done