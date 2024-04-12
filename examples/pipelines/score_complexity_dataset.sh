SCORETYPE="complexity"
DATAPATH="data/deita/tmp.json"
OUTPUTPATH="data/deita/tmp_complexity.json"
MODELPATH="hkust-nlp/deita-complexity-scorer"
SCORER="llama"

python examples/pipelines/score_complexity_dataset.py \
--data_path $DATAPATH \
--output_path $OUTPUTPATH \
--score_type $SCORETYPE \
--scorer $SCORER \
--scorer_name_or_path $MODELPATH