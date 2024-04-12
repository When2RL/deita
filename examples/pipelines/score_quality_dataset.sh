
SCORETYPE="quality"
# DATAPATH="when2rl/UltraFeedback_binarized_cleaned_annotated"
# OUTPUTPATH="data/deita/ultafbk_quality.json"
# OUTPUTPATH="data/deita/ultafbk_quality_mistral.json"
DATAPATH="when2rl/distilabel-intel-orca-dpo-pairs_cleaned_reformatted"
OUTPUTPATH="data/deita/orca-pairs_quality.json"
MODELPATH="hkust-nlp/deita-quality-scorer"  # this is llama architecture
SCORER="llama"
# MODELPATH="mistralai/Mistral-7B-v0.1"
# SCORER="mistral"
GPUINDICES="7"

CUDA_VISIBLE_DEVICES=$GPUINDICES python examples/pipelines/score_complexity_dataset.py \
--data_path $DATAPATH \
--output_path $OUTPUTPATH \
--score_type $SCORETYPE \
--scorer $SCORER \
--scorer_name_or_path $MODELPATH