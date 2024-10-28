#!/bin/bash

# Run the GATGNN model
TL_BACKEND="paddle" python train.py --property bulk-modulus \
                                    --data_src CGCNN \
                                    --num_layers 3 \
                                    --num_neurons 64 \
                                    --num_heads 4 \
                                    --use_hidden_layers True \
                                    --global_attention composition \
                                    --cluster_option fixed \
                                    --concat_comp False \
                                    --train_size 0.8
