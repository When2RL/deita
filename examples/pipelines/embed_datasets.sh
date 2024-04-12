GPUIDX="4,5,6,7"
NUMPROCESS=4
BSZ=1
# DATAPATH="when2rl/UltraFeedback_binarized_cleaned_annotated"
# OUTPUTPATH="data/deita/ultrafbk_embeddings.pkl"
DATAPATH="when2rl/distilabel-intel-orca-dpo-pairs_cleaned_reformatted"
OUTPUTPATH="data/deita/orca-pairs_embeddings.pkl"

CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
--mixed_precision bf16 \
--num_processes $NUMPROCESS \
--num_machines 1 \
examples/pipelines/embed_datasets.py \
--use_flash_attention true \
--data_path $DATAPATH \
--output_path $OUTPUTPATH \
--batch_size_per_device $BSZ
