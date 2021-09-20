#!/bin/bash
## Add parent directory to python path to access lightning_base.py
# export PYTHONPATH="../":"${PYTHONPATH}"

TRAIN_BATCH_SIZE=16
TEST_BATCH_SIZE=4
MAX_LEN=120
BEAM_SIZE=100
NUM_EPOCHS=30

MODEL=t5-base
DATASET=wordnet
DATA_DIR=data/${DATASET}/
T5_OUTPUT_DIR=output/${DATASET}/
T5_GENERAL_OUTPUT_DIR=output/t5_general/${DATASET}/
T5_SPECIFIC_OUTPUT_DIR=output/t5_specific/${DATASET}/
SCORE_DIR=output/evaluation_scores/
SCORE_TYPE=nist

# might need to change permission  eg. chmod +x ./sentence-bleu

#-------------------------------------------------
# train T5, T5-general and T5-specific and generate T5 predictions for validation set and test set

# train T5
export CUDA_VISIBLE_DEVICES=0
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --learning_rate 3e-4 --early_stopping_patience 5 --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --num_train_epochs $NUM_EPOCHS --gpus 1 --do_train --option "" &

# train T5-specific 
export CUDA_VISIBLE_DEVICES=1
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --learning_rate 3e-4 --early_stopping_patience 5 --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_SPECIFIC_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --num_train_epochs $NUM_EPOCHS --gpus 1 --do_train --option "t5_specific" &

# train T5-general 
export CUDA_VISIBLE_DEVICES=2
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --learning_rate 3e-4 --early_stopping_patience 5 --train_batch_size $TRAIN_BATCH_SIZE --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_GENERAL_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --num_train_epochs $NUM_EPOCHS --gpus 1 --do_train --option "t5_general" &
wait

echo "model training finished" > ${SCORE_DIR}progress.txt
# -------------------------------------------------
# generate predictions for validation set and test set

export CUDA_VISIBLE_DEVICES=0
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --gpus 1 --do_predict --test_dataset test_val --num_beams $BEAM_SIZE --option "" &  

export CUDA_VISIBLE_DEVICES=1
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --gpus 1 --do_predict --num_beams $BEAM_SIZE --option "" &
wait

cp ${T5_OUTPUT_DIR}val_predictions.txt ${DATA_DIR}val.forward
cp ${T5_OUTPUT_DIR}test_predictions.txt ${DATA_DIR}test.forward

echo "predictions generated" > ${SCORE_DIR}progress.txt
# -------------------------------------------------
# calculate scores of predictions generated
# for mode mose bleu, you need to run calculate_scores.sh

python calculate_scores.py --pred_dir $DATA_DIR --data_dir $DATA_DIR --output_dir "${SCORE_DIR}${DATASET}" --beam_sz $BEAM_SIZE --type_path "val" --mode $SCORE_TYPE &

python calculate_scores.py --pred_dir $DATA_DIR --data_dir $DATA_DIR --output_dir "${SCORE_DIR}${DATASET}" --beam_sz $BEAM_SIZE --type_path "test" --mode $SCORE_TYPE &

bash calculate_scores.sh

echo "scores of predictions calculated" >> ${SCORE_DIR}progress.txt
# -------------------------------------------------
# calculate generation score, generate score, specific score on validation set

export CUDA_VISIBLE_DEVICES=0
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --do_predict --test_dataset test_val --gpus 1 --option "forward" &

export CUDA_VISIBLE_DEVICES=1
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_SPECIFIC_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --do_predict --test_dataset test_val --gpus 1 --option "t5_specific" &

export CUDA_VISIBLE_DEVICES=2
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_GENERAL_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --do_predict --test_dataset test_val --gpus 1 --option "t5_general" &
wait

echo "re-ranking scores on test set calculated" >> ${SCORE_DIR}progress.txt
# -------------------------------------------------
# calculate generation score, generate score, specific score on test set

export CUDA_VISIBLE_DEVICES=0
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --gpus 1 --do_predict --option "forward" &

export CUDA_VISIBLE_DEVICES=2
python finetune.py --model_name_or_path $MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_SPECIFIC_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --do_predict --gpus 1 --option "t5_specific" &

export CUDA_VISIBLE_DEVICES=1
python finetune.py --model_name_or_path　$MODEL --data_dir $DATA_DIR --eval_batch_size $TEST_BATCH_SIZE --output_dir $T5_GENERAL_OUTPUT_DIR --max_source_length $MAX_LEN --max_target_length $MAX_LEN --do_predict --gpus 1 --option "t5_general" &
wait

echo "re-ranking scores on validation set calculated" >> ${SCORE_DIR}progress.txt
# -------------------------------------------------
# search for weights on validation set and apply to test set
# predictions and score will be written to directory specified by parameter output_dir

python search_weights.py --dataset_name $DATASET --data_dir $DATA_DIR --f_dir $T5_OUTPUT_DIR --g_dir $T5_GENERAL_OUTPUT_DIR --s_dir $T5_SPECIFIC_OUTPUT_DIR --evaluation_score_path $SCORE_DIR --output_dir "" --beam_sz $BEAM_SIZE --score_type $SCORE_TYPE &

python search_weights.py --dataset_name $DATASET --data_dir $DATA_DIR --f_dir $T5_OUTPUT_DIR --g_dir $T5_GENERAL_OUTPUT_DIR --s_dir $T5_SPECIFIC_OUTPUT_DIR --evaluation_score_path $SCORE_DIR --output_dir="" --beam_sz $BEAM_SIZE --score_type "mose" &
wait

head -1 "${DATASET}_${SCORE_TYPE}_best_predictions.txt"
head -1 "${DATASET}_mose_best_predictions.txt"