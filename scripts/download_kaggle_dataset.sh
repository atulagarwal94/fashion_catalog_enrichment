#!/usr/bin/env bash
set -e
mkdir -p data/raw
echo "Downloading Fashion Product Images Small dataset from Kaggle..."
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/raw --unzip
echo "Done. Expected files: data/raw/styles.csv and data/raw/images/"
