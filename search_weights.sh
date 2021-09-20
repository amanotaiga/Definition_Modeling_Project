
python search_weights.py \
--dataset_name="wordnet" \
--data_dir="transformers1/examples/seq2seq/cnn_tiny/wordnet/" \
--f_dir="transformers1/examples/seq2seq/bart_utest_output/" \
--g_dir="transformers1/examples/seq2seq/bart_utest_output/without_context/" \
--s_dir="transformers1/examples/seq2seq/bart_utest_output/reversed_context/" \
--evaluation_score_path="transformers1/examples/seq2seq/bart_utest_output/bleus/" \
--output_dir="transformers1/examples/seq2seq/bart_utest_output/bleus/testing.txt" \
--beam_sz=100 \
--score_type="nist" \


# python search_weights.py \
# --dataset_name="japanese_wiki" \
# --data_dir="transformers1/examples/seq2seq/cnn_tiny/japanese_wiki/" \
# --f_dir="transformers1/examples/seq2seq/bart_utest_output/" \
# --g_dir="transformers1/examples/seq2seq/bart_utest_output/without_context/" \
# --s_dir="transformers1/examples/seq2seq/bart_utest_output/reversed_context/" \
# --evaluation_score_path="transformers1/examples/seq2seq/bart_utest_output/bleus/" \
# --output_dir="transformers1/examples/seq2seq/bart_utest_output/bleus/testing.txt" \
# --beam_sz=100 \
# --score_type="nist"