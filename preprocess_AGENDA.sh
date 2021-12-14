#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_AGENDA.sh <dataset_folder>"
  exit 2
fi

processed_data_folder='graph2text/data/agenda'
mkdir -p ${processed_data_folder}

#python preprocess/generate_input_agenda.py ${1} ${processed_data_folder}

python graph2text/preprocess.py -train_src ${processed_data_folder}/training-nodes.txt \
                       -train_graph ${processed_data_folder}/training-graph.txt \
                       -train_tgt ${processed_data_folder}/training-surface.txt \
                       -train_plan ${processed_data_folder}/training-plans.txt \
                       -train_tree ${processed_data_folder}/training-trees.txt \
                       -valid_src ${processed_data_folder}/dev-nodes.txt  \
                       -valid_graph ${processed_data_folder}/dev-graph.txt  \
                       -valid_tgt ${processed_data_folder}/dev-surface.txt \
                       -valid_plan ${processed_data_folder}/dev-plans.txt \
                       -valid_tree ${processed_data_folder}/dev-trees.txt \
                       -save_data ${processed_data_folder}/agenda \
                       -src_vocab_size 30000 \
                       -tgt_vocab_size 30000 \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -dynamic_dict \
                       -share_vocab \
                       -overwrite

