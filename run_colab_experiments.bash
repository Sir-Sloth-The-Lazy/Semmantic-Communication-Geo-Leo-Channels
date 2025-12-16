#!/bin/bash

# Define common parameters
# Adjust these based on your specific needs
DATA_SIZE=512
SNR_DB=10
BLOCK_TYPES="CCCVVVCC"
EPOCHS=300
FILTERS="256 256 256 256 256 256"
REPETITIONS="1 1 3 3 1 1"

echo "========================================================"
echo "Starting Sequential Training on Google Colab (A100)"
echo "========================================================"

# 1. Mixed Satellite Model
echo ""
echo "[1/3] Training Mixed Satellite Model (LEO + Probabilistic GEO)..."
python train_dist.py $DATA_SIZE Satellite $SNR_DB $BLOCK_TYPES "experiment_mixed_sat" $EPOCHS \
    --filters $FILTERS \
    --repetitions $REPETITIONS

# 2. LEO Model
echo ""
echo "[2/3] Training LEO Satellite Model (Strictly Low Earth Orbit)..."
python train_dist.py $DATA_SIZE LEO $SNR_DB $BLOCK_TYPES "experiment_leo_sat" $EPOCHS \
    --filters $FILTERS \
    --repetitions $REPETITIONS

# 3. GEO Model
echo ""
echo "[3/3] Training GEO Satellite Model (Strictly Geostationary)..."
python train_dist.py $DATA_SIZE GEO $SNR_DB $BLOCK_TYPES "experiment_geo_sat" $EPOCHS \
    --filters $FILTERS \
    --repetitions $REPETITIONS

echo ""
echo "========================================================"
echo "All Experiments Completed Successfully!"
echo "Check 'ckpt/' and 'logs/' for results."
echo "========================================================"
