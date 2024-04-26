# Imbalanced-LLM

Classification of Imbalanced Data with LLM (Large Language Model)

This project code uses [TabLLM](https://github.com/clinicalml/TabLLM) and [T-few](https://github.com/r-three/t-few) project and its paper.

As an Undergraduate Researcher, I did not had the luxury to have over 30GB GPU, and did not want to spend money on Colab. So I had to modify lots of code and versionings to make it work on free tier Colab, and even locally with smaller LLMs.

If there are questions, please leave an issue on github to talk more about it, or email me at contactjameshan@gmail.com

## Version
- This code is ran on Google Colab Free Tier. It will follow those versions.
- Used Python Black Formatter

## Folders

- `/.old`: Old attempt for Imbalanced LLM. Testing Idea.
- `/bin`: Shell code to run the project
- `/configs`: Configuration Data, related to `/src/utils/Config.py`
- `/Datasets`: Raw csv datasets (Not included, go to [TabLLM Project](https://github.com/clinicalml/TabLLM))
- `/Datasets-serialized`: Serialized datasets (Not included, go to [TabLLM Project](https://github.com/clinicalml/TabLLM))
- `/exp_out`: Your Train result (Not included)
- `/pretrained_checkpoints`: Saved Model (Not included)
  -  If using model T0 or T0_3b, get the file from [TabLLM Project](https://github.com/clinicalml/TabLLM), turn on `load_model()` in `EncoderDecoder()`  and add file
- `/src`: Your Source
- `/templates`:

# Testing
> This tutorial is modified from [TabLLM Project](https://github.com/clinicalml/TabLLM).

We will use Google Colab Free Tier to run.
## How to add your own Dataset
For my case, we will use [stroke-prediction-dataset/data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

### 1. Serialize Dataset
- Run Make_Datset.ipynb
Or
- `create_external_datasets.py --dataset stroke`
### 2. Add files
- Go to `evaluate_external_dataset` and add your dataset name on `args_datasets` variable
- Make a new file called `template_<datasetName>` on `templates` folder. Use other templates as reference.
- Go to `bin/few-shot-pretrained-100k.sh` and add your dataset on `for dataset in <dataSetName>`

## Train/Fine Tune
- Run TabLLM.ipynb

## Get Result
> For Stroke Dataset
- `python src/scripts/get_result_table.py -e t5_\* -d stroke`
