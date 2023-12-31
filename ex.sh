#/bin/bash

# CIL CONFIG
NOTE="ocs_final_blurry_rescue"
MODE="ocs"
K_COEFF="4"
TEMPERATURE="0.125"
TRANSFORM_ON_GPU="--transform_on_gpu"
#TRANSFORM_ON_GPU=""
N_WORKER=1
FUTURE_STEPS=1
EVAL_N_WORKER=2
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.25
SEEDS="1 2 3"
KNN_TOP_K="3"
SELECT_CRITERION="softmax"
LOSS_CRITERION="DR"
SOFTMAX_TEMPERATURE=1.0
KNN_SIGMA=0.7
RESIDUAL_NUM=50
RESIDUAL_NUM_THRESHOLD=10
CURRENT_FEATURE_NUM=50
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=1
SIGMA=10
REPEAT=1
INIT_CLS=100
USE_AMP="--use_amp"
NUM_EVAL_CLASS=10
NUM_CLASS=10
DISTILL_COEFF=0.99
DISTILL_BETA=0.5
DISTILL_THRESHOLD=0.5
DISTILL_STRATEGY="classwise_difference" # naive, classwise, classwise_difference 
RESIDUAL_STRATEGY="none" # prob, none
OOD_STRATEGY="none" # cutmix, rotate, none
OOD_NUM_SAMPLES=16
SCL_COEFF=0.01
#TRANSFORMS=['randaug', 'cutmix']

### DISTILLATION ###
#USE_FEATURE_DISTILLATION="--use_feature_distillation"
USE_FEATURE_DISTILLATION=""

### STORING PICKLE ###
#STORE_PICKLE="--store_pickle"
STORE_PICKLE=""

### RESIDUAL ###
#USE_RESIDUAL="--use_residual"
USE_RESIDUAL=""

#RESIDUAL_WARM_UP="--use_residual_warmup"
RESIDUAL_WARM_UP=""

#RESIDUAL_UNIQUE="--use_residual_unique"
RESIDUAL_UNIQUE=""

#MODIFIED_KNN="--use_modified_knn"
MODIFIED_KNN=""

#PATCH_PERMUATION="--use_patch_permutation"
PATCH_PERMUATION=""

#REGULARIZATION="--use_synthetic_regularization"
REGULARIZATION=""


if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000
    N_SMP_CLS="3" K="3" MIR_CANDS=100
    CANDIDATE_SIZE=200 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167
    N_SMP_CLS="3" K="3" MIR_CANDS=800
    CANDIDATE_SIZE=1000 VAL_SIZE=2
    MODEL_NAME="resnet18" EVAL_PERIOD=8000 F_PERIOD=200000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

if [ "$MODE" == "ocs" ]; then
    BATCHSIZE=$((2*BATCHSIZE))
    EVAL_PERIOD=$((2*EVAL_PERIOD))
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0 nohup python main_new.py --mode $MODE --residual_strategy $RESIDUAL_STRATEGY $RESIDUAL_UNIQUE \
    --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE --ood_strategy $OOD_STRATEGY --scl_coeff $SCL_COEFF \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --samples_per_task 10000 --residual_num $RESIDUAL_NUM $RESIDUAL_WARM_UP $MODIFIED_KNN --ood_num_samples $OOD_NUM_SAMPLES \
    --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE --num_eval_class $NUM_EVAL_CLASS --num_class $NUM_CLASS --residual_num_threshold $RESIDUAL_NUM_THRESHOLD \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME --softmax_temperature $SOFTMAX_TEMPERATURE $PATCH_PERMUATION $REGULARIZATION \
    --lr $LR --batchsize $BATCHSIZE --mir_cands $MIR_CANDS $STORE_PICKLE --knn_top_k $KNN_TOP_K --select_criterion $SELECT_CRITERION $USE_RESIDUAL $USE_FEATURE_DISTILLATION \
    --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER --knn_sigma $KNN_SIGMA --distill_coeff $DISTILL_COEFF --distill_beta $DISTILL_BETA --distill_threshold $DISTILL_THRESHOLD --distill_strategy $DISTILL_STRATEGY --current_feature_num $CURRENT_FEATURE_NUM \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE --n_tasks 5 &
done
