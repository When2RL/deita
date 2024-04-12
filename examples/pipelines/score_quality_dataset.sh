
SCORETYPE="quality"
DATAPATH="data/deita/tmp_complexity.json" # PATH/TO/DATASET
OUTPUTPATH="data/deita/tmp_complexity_quality.json"  # PATH/TO/OUTPUTS
MODELPATH="hkust-nlp/deita-quality-scorer"    # PATH/TO/MODEL
SCORER="llama"
ISVLLM=true
GPUINDICES="0"

CUDA_VISIBLE_DEVICES=$GPUINDICES python examples/pipelines/score_complexity_dataset.py \
--data_path $DATAPATH \
--output_path $OUTPUTPATH \
--score_type $SCORETYPE \
--scorer $SCORER \
--scorer_name_or_path $MODELPATH \
--is_vllm $ISVLLM