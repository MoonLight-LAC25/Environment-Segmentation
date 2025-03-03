#!/usr/bin/env bash

python make_single_channel.py

python voc_annotation.py

python train.py