#!/bin/bash

if [ "$#" -lt 5 ]; then
  echo "./decode_AGENDA.sh <gpuid> <model> <nodes-file> <graph-file> <plan-file> <tree-file> <output>"
  exit 2
fi

GPUID=$1
MODEL=$2
NODES_FILE=$3
GRAPH_FILE=$4
PLAN_FILE=$5
TREE_FILE=$6
OUTPUT=$7

export CUDA_VISIBLE_DEVICES=${GPUID}
export OMP_NUM_THREADS=10

python -u graph2text/translate.py -model ${MODEL} \
-src ${NODES_FILE} \
-graph ${GRAPH_FILE} \
-plan ${PLAN_FILE} \
-tree ${TREE_FILE} \
-output ${OUTPUT} \
-beam_size 5 \
-share_vocab \
-min_length 0 \
-max_length 430 \
-length_penalty wu \
-alpha 5 \
-verbose \
-batch_size 80 \
-gpu 0

cat ${OUTPUT} | sed -r 's/(@@ )|(@@ ?$)//g' > "${OUTPUT}_proc.txt"
mv "${OUTPUT}_proc.txt" ${OUTPUT}