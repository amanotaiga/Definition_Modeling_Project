DATASET=oxford
DATA_DIR=data/oxford/
SCORE_DIR=output/scores/
SCORE_TYPE=mose
BEAM_SIZE=100

# -------------------------------------------------
# calculate mose sentence bleus spliting data into n=5 splits (to accelerate) and combine them into one file

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp1/" --mode=$SCORE_TYPE --c_range="0,0.1" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp2/" --mode=$SCORE_TYPE --c_range="0.1,0.2" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp3/" --mode=$SCORE_TYPE --c_range="0.2,0.3" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp4/" --mode=$SCORE_TYPE --c_range="0.3,0.4" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp5/" --mode=$SCORE_TYPE --c_range="0.4,0.5" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp6/" --mode=$SCORE_TYPE --c_range="0.5,0.6" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp7/" --mode=$SCORE_TYPE --c_range="0.6,0.7" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp8/" --mode=$SCORE_TYPE --c_range="0.7,0.8" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp9/" --mode=$SCORE_TYPE --c_range="0.8,0.9" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="val" --tmp_dir="tmp10/" --mode=$SCORE_TYPE --c_range="0.9,1.0" &
wait

cat "${SCORE_DIR}${DATASET}_val_mose_tmp"*.txt > "${SCORE_DIR}${DATASET}_val_mose.txt"
rm "${SCORE_DIR}${DATASET}_val_mose_tmp"*.txt

# -------------------------------------------------
# calculate mose sentence bleus spliting data into n=5 splits (to accelerate) and combine them into one file

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp1/" --mode=$SCORE_TYPE --c_range="0,0.1" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp2/" --mode=$SCORE_TYPE --c_range="0.1,0.2" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp3/" --mode=$SCORE_TYPE --c_range="0.2,0.3" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp4/" --mode=$SCORE_TYPE --c_range="0.3,0.4" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp5/" --mode=$SCORE_TYPE --c_range="0.4,0.5" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp6/" --mode=$SCORE_TYPE --c_range="0.5,0.6" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp7/" --mode=$SCORE_TYPE --c_range="0.6,0.7" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp8/" --mode=$SCORE_TYPE --c_range="0.7,0.8" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp9/" --mode=$SCORE_TYPE --c_range="0.8,0.9" &

python calculate_scores.py --pred_dir=$DATA_DIR --data_dir=$DATA_DIR --output_dir="${SCORE_DIR}${DATASET}" --beam_sz=$BEAM_SIZE --type_path="test" --tmp_dir="tmp10/" --mode=$SCORE_TYPE --c_range="0.9,1.0" &
wait

cat "${SCORE_DIR}${DATASET}_test_mose_tmp"*.txt > "${SCORE_DIR}${DATASET}_test_mose.txt"
rm "${SCORE_DIR}${DATASET}_test_mose_tmp"*.txt