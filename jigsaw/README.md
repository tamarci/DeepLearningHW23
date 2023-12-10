# Linear benchmark evaluation of the Jigsaw task

## Motivation

...

## Running with Docker (recommended)

Change your working directory to the current (`jigsaw`) folder in the repo. Then to build the image and run the container:

```
docker build -t {IMAGE_NAME} .
docker run --rm --gpus all -it jigsaw
```
This leaves you in the interactive terminal of the environment. To run a baseline version of the pretraining routines, the linear benchmark evaluation, and the hyperparameter optimization run either
```
/usr/bin/python3 train_and_evaluate.py pretrain_and_benchmark
```
or
```
/usr/bin/python3 train_and_evaluate.py hyperopt
```
Additionally, providing `--model_path` (default: `models/jigsaw_10_1024.pth`) lets you specify where to save the feature extractor portion of the model.


Running either will download the datasets (the validation set of the Places365 - not the train split, because that is cca. 100 GB, the Tiny Imagenet dataset) into the container's local file system, and a prompt will ask for your API key at WandB to track runs. After successful authentication, the appropriate action starts.

The `pretrain_and_benchmark` action (it is advised to run this first, to have a saved feature extractor on which the hyperopt step is based):
1. Trains a model on the self-supervised jigsaw task, using a 70-15-15 train-val-test split of the Places365 validation dataset, while logging the results to WandB.
2. Saves the feature extractor portion of the model
3. Runs the linear benchmark evaluation, by loading the feature extractor, placing a single classifier linear layer on top of it, and training only the linear weights for the classification task. This is performed on the Tiny ImageNet dataset, that provides 200 image classes.

The `hyperopt` action (it is advised to run after the above) performs hyperparameter optimization on step 3 from above, attempting to pinpoint the optimal learning rate and batch parameters to run the linear benchmark evaluation upon, thus assessing the pretrained feature extractor "at its best". Due to hardware and time constraints, we only ran the hyperparameter optimization os this step, to demonstrate its capabilities, but based on the code, it can be easily expanded to step 1, by trying different configurations of the model itself (setting global variables `CONV{i}_C`, `FC{i}` etc.). The optimization itself is done via WandB Sweeps.


## How to configure

Going beyond the basic configuration of the functionalities described above, various model and training parameters can be customized by editing the global variabes of `train_and_evaluate.py` (right after the imports) before running the Docker build. This includes the sizes of the convolutional backbone, the number of neurons in the fully connected layers, but the pretrain task too, to some extent (the numer of tile divisions, e.g. 2 by 2, 3 by 3 - the default - can be set via `N_TILES`). We chose this customization option of editing global variables over cluttering the CLI interface of the script.

File and folder dependencies (the files and folders that should exist before starting the script) are either readily available in the repository, or created during the Docker build. File output (where to save the weights of the feature extractor) can be set via `--model_path`, datasets will also be downloaded locally to the conatiner's filesystem on first run.

The basic usage might run out of CPU or GPU memory, or overload the CPU. In this case, performance parameters (like the `num_workers`, `batch_size`) should be overwritten in the appropriate functions (`pretrain_and_save`, `load_feature_extractor_and_evaluate`, `linear_benchmark_evaluation_with_sweeps`, `fit_sweep`, generally where data and model objects are being instantiated).