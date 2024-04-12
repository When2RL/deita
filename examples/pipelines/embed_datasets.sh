GPUIDX="4,5,6,7"
NUMPROCESS=4
BSZ=1
DATAPATH="data/deita/tmp_complexity_quality.json"
OUTPUTPATH="data/deita/tmp_embeddings.pkl"

CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
--mixed_precision bf16 \
--num_processes $NUMPROCESS \
--num_machines 1 \
examples/pipelines/embed_datasets.py \
--use_flash_attention true \
--data_path $DATAPATH \
--output_path $OUTPUTPATH \
--batch_size_per_device $BSZ
