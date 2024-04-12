GPUIDX="7"
NUMGPUS=$(echo $GPUIDX | awk -F',' '{print NF}')
DATAPATH="data/deita/tmp_complexity_quality.json"
OTHERDATA="data/deita/tmp_embeddings.pkl"
OUTPUTPATH="data/deita/tmp_deita.csv"   # only full_ids
THETA=0.9
DATASIZE=10
BSZ=1

CUDA_VISIBLE_DEVICES=$GPUIDX python examples/pipelines/combined_filter.py \
--data_path $DATAPATH \
--other_data_path $OTHERDATA \
--output_path $OUTPUTPATH \
--threshold $THETA \
--data_size $DATASIZE \
--device 0
