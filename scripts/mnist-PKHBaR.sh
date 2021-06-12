#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset=mnist
model=lenet3

st=irregular
preCe=${dataset}_${model}_xw_1_lx_0_ly_0_adv.pt
preHBaR=${dataset}_${model}_xw_1_lx_0.004_ly_0.001_linear_adv-0100.pt
distillModel=${preHBaR}

xw=0
lx=0.001
ly=0.005

pr=(0.5 0.75 0.875 0.9375)

for ((i=0;i<${#pr[@]};i++)) do
    for ((j=0;j<${#lx[@]};j++)) do
        save_name=prune_${pr[i]}_${st}_xw_${xw}_lx_${lx[j]}_ly_${ly}_hb_distillSgd_hbSgd
        run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml \
            -slmo -pr ${pr[i]} -st $st -xw $xw \
            -op sgd --lr-scheduler default -lx ${lx[j]} -ly ${ly} \
            -reopt sgd -ree 20 -relr 0.0001 -relx ${lx[j]} -rely ${ly} \
            -mf ${dataset}_${model}_${save_name}.pt \
            -lm $preHBaR \
            --distill --distill_loss kl --distill_model_path ${distillModel} --distill_temp 30 --distill_alpha 1 \
            >logs/${dataset}/${model}/${save_name}.out
    done
done