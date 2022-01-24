
dataset=mnist
model=lenet3

st=irregular
prePGD=${dataset}_${model}_xw_1_lx_0.004_ly_0.001_linear_adv-0100.pt


pretrain_prune_finetune_hb_mixup() {
    xw=0
    lx=0.0002
    ly=0.00005
    pr=$1
    premodel=$4
    distillModel=/home/tong/MIBottleneck/MIBottleneck-Pruning/assets/models/${prePGD}
    save_name=prune_${pr}_${st}_teacher_PGD_student_PGD_mix_$2
    run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml \
        -ep 50 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.0005 -bs 128 \
        -reopt adam --lr-scheduler default -ree 100 -relr 0.001 -relx ${lx} -rely ${ly} \
        -mf ${dataset}_${model}_${save_name}.pt \
        -lm $premodel \
        --distill --distill_loss kl --distill_model_path ${distillModel} --distill_temp 30 --distill_alpha 10 \
        -adv adv --mix_ratio $2 --device $3 \
        >logs/${dataset}/${save_name}_v4.out
}


# mixed training
pretrain_prune_finetune_hb_mixup 0.75 0 "cuda:0" ${prePGD};
pretrain_prune_finetune_hb_mixup 0.75 0.2 "cuda:0" ${prePGD};
pretrain_prune_finetune_hb_mixup 0.75 0.4 "cuda:0" ${prePGD};
pretrain_prune_finetune_hb_mixup 0.75 0.6 "cuda:0" ${prePGD};
pretrain_prune_finetune_hb_mixup 0.75 0.8 "cuda:0" ${prePGD};
pretrain_prune_finetune_hb_mixup 0.75 1 "cuda:0" ${prePGD};
