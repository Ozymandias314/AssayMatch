# AssayMatch
This repository contains code for AssayMatch: Language Guided Data Selection for Molecular Activity with Data Attribution.

To install 
```
#create a new environment
conda create -n AssayMatch python=3.11
conda activate AssayMatch
python -m pip install -r requirements.txt
```

## Code Description
```
AssayMatch/
├── assaymatch/
│   ├── step1_prepare_trak.py #Compute TRAK scores and aggregate
│   ├── step2_prepare_finetune.py #Finetune embeddings with TRAK derived signal
│   └── step3_generate_data.py #Greedily select datasets with TRAK scores
├── scripts/
│   ├── dispatcher_multi.py #AssayMatch generates config files that are run other scripts through this dispatcher
│   └── #... other scripts
├── benchmark/
│   └── run1/ #benchmarking runs go here
├── smiles-transformer/ (adapted from Shion Honda, 2019 under the MIT License)
├── README.md
├── requirements.txt
└── benchmark.sh #run this to do a new benchmarking trial
```


This project includes code adapted from SmilesTransformer (Shion Honda, 2019), used under the MIT License.
