#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# basic exp: 
# 1) xentropy only
# 2) hisc + xentropy

dataset=cifar100
model=wideresnet

st=irregular

preCe=${dataset}_${model}_xw_1_lx_0_ly_0_adv.pt
preHBaR=${dataset}_${model}_xw_1_lx_0.0005_ly_0.005_adv.pt
distillModel=${preHBaR}

xw=0
lx=0.000001
ly=0.000005

pr=(0.5 0.75 0.875 0.9375)

for ((i=0;i<${#pr[@]};i++)) do
    for ((j=0;j<${#lx[@]};j++)) do
        save_name=prune_${pr[i]}_${st}_xw_${xw}_lx_${lx[j]}_ly_${ly}_hb_distillSgd_hbSgd
        run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml \
            -slmo -pr ${pr[i]} -st $st -xw $xw \
            -op sgd --lr-scheduler default -lx ${lx[j]} -ly ${ly} \
            -reopt sgd -ree 100 -relr 0.01 -relx ${lx[j]} -rely ${ly} \
            -mf ${dataset}_${model}_${save_name}.pt \
            -lm $preHBaR \
            --distill --distill_loss kl --distill_model_path ${distillModel} --distill_temp 30 --distill_alpha 1 \
            >logs/${dataset}/${model}/${save_name}.out
    done
done