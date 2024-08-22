#!/usr/bin/env bash

cd ..

python test.py \
--model_name duel \
--dataset_name Thumos14reduced \
--path_dataset /data_SSD1/cmy/CO2-THUMOS-14 \
--num_class 20 \
--use_model CO2 \
--without_wandb
