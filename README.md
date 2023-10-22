# Self-supervised learning “in-the-wild"

## Team: ChatGPT suggest a teamname

## Authors
- [**István Péter (TCTYJW)**](https://github.com/)
- [**Márton Tárnok (GGDVB2)**](https://github.com/tamarci)
- [**Miklós Bartos-Elekes (YQ9MV3)**](https://github.com/)

## Description
This is a university project for the subject [**Deep Learning(VITMMA19)**](https://portal.vik.bme.hu/kepzes/targyak/VITMMA19/). 

Many of the popular self-supervised learning (SSL) methods are tailored for ImageNet and may underperform on unfamiliar datasets. To understand these limitations, we explore and choose two open-source SSL methods with ImageNet-pretrained models. We then pretrain these models on 1-2 non-ImageNet classification datasets within our computing capacity. We evaluate the models using the linear benchmark on ImageNet subsets and compare results, considering factors like computational costs and training times.

## Related papers
 - https://arxiv.org/abs/2103.01988
 - https://arxiv.org/abs/2103.13559
 - https://proceedings.neurips.cc/paper/2020/hash/22f791da07b0d8a2504c2537c560001c-Abstract.html

## Related repos
 - https://github.com/fastai/imagenette
 - https://github.com/lightly-ai/lightly
 - https://github.com/vturrisi/solo-learn

## Datasets

 - https://www.kaggle.com/datasets/ambityga/imagenet100
 - https://huggingface.co/datasets/zh-plus/tiny-imagenet

## Description of the files and folders in the repo

**preprocessing folder**
 - Purpose: Handles initial stages of data processing.
 - Data cleaning and transformation.
 - Definition of dataloaders and datasets for model training and evaluation.
   
**notebooks**
 - Purpose: Contains Jupyter notebooks for various tasks.
 - Data visualization for insights into dataset characteristics.
 - Quick and interactive experiments or prototyping.
   
**Dockerfile**
 - Purpose: Script for creating a Docker image.
 - Automates setting up the virtual environment.
 - Ensures consistency and reproducibility.
   
**requirements.txt**
 - Purpose: Lists repository dependencies.
 - Ensures necessary Python packages are installed.
 - Simplifies environment setup for users.
   
**train.py**
 - Purpose: Script for model training initiation.
 - Configures model training parameters.
 - Manages the training process and saves trained models.
   
**predict.py**
 - Purpose: Utility script for model predictions.
 - Loads trained models and outputs predictions based on input data.
   
**models**
 - Purpose: Storage for model architectures.
 - Preserves and organizes various model structures.
 - Provides easy access for revisiting or reloading specific models.
   
**results**
 - Purpose: Repository section for outcomes and model outputs.
 - Displays visualized results and insights into model behavior.
 - Storage for plots, figures, and related visual artifacts.
   
**utils**
 - Purpose: A directory for utility functions and scripts.
 - Contains miscellaneous helper functions that can be used across different parts of the repository.
 - Ensures code reusability and organization.

 ## How to run

 After pulling the project, first make sure you have both Docker and [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. If you install it just before building and running the container, make sure to restart the Docker daemon with
 ```
 sudo systemctl restart docker
 ```
 (Assuming you work in a Linux environment with `systemd`. For other use cases, see the Docker documentation for your platform).
 Some errors down the road can also be fixed by restarting the Docker runtime.
 
 In the repo's directory, run:
 ```
 docker build -t {IMAGE_NAME} .
 ```
 where you can specify `{IMAGE_NAME}`, then:
 ```
 docker run --rm --gpus all -it {IMAGE_NAME}
 ```
 to enter in the interactive terminal after running the commands in the [Dockerfile](Dockerfile). 

By specifying port forwarding (`-p {HOST_PORT}:{CONTAINER_PORT}`) from the container port 8888 (used by `jupyter notebook`) to the host port of your liking, you can run the `jupyter notebook` command inside the terminal and access the notebook itself at `http://localhost:{PORT}`, where you can run the preprocess notebook.
