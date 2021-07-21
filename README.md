when-do-reading-comprehension-models-learn
==============================

Repository for MSc Machine Learning thesis titled 'When do Reading Comprehension Models Learn?'

Project Organization
------------

    ├── LICENSE    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.        
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data    
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling    
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions      
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations    


Set-up
------------

```bash
# Create environment 
conda create -n rclearn python=3.7 -y
conda activate rclearn

# Clone own copy of `huggingface/transformers` (as of 27/04/21):
cd ..
git clone https://github.com/stevieg3/mytransformers.git

# Install library from source
cd mytransformers
pip install -e .

# Install PyTorch - use conda for CUDA support
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install other packages
cd ../when-do-reading-comprehension-models-learn
pip install -r requirements.txt

# Download Adversarial QA files (to construct custom dataset combinations)
wget -P data/external/ "https://adversarialqa.github.io/data/aqa_v1.0.zip"
unzip data/external/aqa_v1.0.zip -d data/external/aqa_v1.0
rm data/external/aqa_v1.0.zip

# Download Stanford CoreNLP models
wget -P data/external/ "https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip"
unzip data/external/stanford-ner-4.2.0.zip -d data/external/stanford-ner-4.2.0
rm data/external/stanford-ner-4.2.0.zip

wget -P data/external/ "https://nlp.stanford.edu/software/stanford-tagger-4.2.0.zip"
unzip data/external/stanford-tagger-4.2.0.zip -d data/external/stanford-tagger-4.2.0
rm data/external/stanford-tagger-4.2.0.zip
```

Run tests
------------
```bash
python -m unittest
```

Generate SQuAD categories
------------
```bash
# SQuAD 1
python src/analysis/squad_categorisation.py --squad_version 1 --split train
python src/analysis/squad_categorisation.py --squad_version 1 --split validation
# SQuAD 2
python src/analysis/squad_categorisation.py --squad_version 2 --split train
python src/analysis/squad_categorisation.py --squad_version 2 --split validation
```

Create weighted combination datasets
------------
```bash
# 2 copies of SQuAD v1.1 and 3 copies of Adversarial QA dBERT:
python src/data/generate_dataset_combination.py --num_squad 2 --num_adversarial 3 --adversarial_model bert
```

--------