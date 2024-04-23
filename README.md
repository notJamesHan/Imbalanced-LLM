# Imbalanced-LLM

Classification of Imbalanced Data with LLM

## Folders

- `old/brev`: Tried to follow [this youtube tutorial](https://youtu.be/ztPoCymwIp0?feature=shared) to fine tune LLM.
- `old/setfit`: Tried to use [setfit](https://github.com/huggingface/setfit), created by Hugging Face.
  

## Datasets
>  I have not included the dataset due to the file size.

- Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Heart Attack: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

## TabLLM needed packages
- Datasets
- Datasets-serialized
- pretrained_checkpoints (Not using, if using T0_3b, turn on load_model() and add file)

## Solve Error

1. use python3.8 for t-few conda env
2. Run this in Linux
3. pip install --upgrade "protobuf<=3.20.1"
4. Download HuggingFace Cli to use the Token

`AttributeError: module 'distutils' has no attribute 'version'`
5. [Fix](https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462)
6. `pip install typing-extensions` dropped support for 3.8, use 3.10


## This is for colab based.
```
conda create -n tfew python==3.10
conda activate tfew

pip install fsspec==2021.05.0 # exists in colab
pip install urllib3==1.26.6 # exists in colab
pip install importlib-metadata==4.13.0 # exists in colab
pip install scikit-learn # exists in colab 

pip install --use-deprecated=legacy-resolver  -r requirements.txt


# If promptsource failed
!pip install git+https://github.com/bigscience-workshop/promptsource.git
```

## TODO
- TypeError: EncoderDecoder.on_validation_epoch_end() missing 1 required positional argument: 'outputs'
- Open Log