# Do Topological Characteristics Help in Knowledge Distillation?
## Accepted in ICML 2024

[[Paper]](💥)

![architecture.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/249207a0-adf7-4d67-a185-e3ab4a927f28/architecture.png)

> Do Topological Characteristics Help in Knowledge Distillation?

International Conference on Machine Learning (ICML) 2024
Jungeun Kim*,  Junwon You*, Dongjin Lee*, Ha Young Kim, Jae-Hun Jung
Yonsei University & POSTECH
> 

## Abstract

Knowledge distillation (KD) aims to transfer knowledge from larger (teacher) to smaller (student) networks. Previous studies focus on point-to-point or pairwise relationships in embedding features as knowledge and struggle to efficiently transfer relationships of complex latent spaces. To tackle this issue, we propose a novel KD method called TopKD, which considers the global topology of the latent spaces. We define global topology knowledge using the persistence diagram (PD) that captures comprehensive geometric structures such as shape of distribution, multiscale structure and connectivity, and the topology distillation loss for teaching this knowledge. To make the PD transferable within reasonable computational time, we employ approximated persistence images of PDs. Through experiments, we support the benefits of using global topology as knowledge and demonstrate the potential of TopKD.

## Installation

### Environment

```bash
conda env create -f environment.yml
conda activate topkd
```

### Create data directory

```bash
mkdir data
```

## Training

*** Please modify the shell script files to match your experimental environment. ***

### Training teacher network

First, train teacher network to extract persistence diagram (PD) of teacher embedding features and distill the knowledge.

```bash
sh train_teacher.sh
```

### Generate point cloud data (PCD) and persistence images (PI)

Next, generate PI from PCD to integrate PD to DNN. Here, PCD is embedding features of teacher network and PI is an vectorization of PD. 

```bash
python generate_PCD_PI.py
```

### Training RipsNet

During training student network, accurately calculating PDs and PIs for each batch requires heavy computational demands. Thus, we utilize [RipsNet](https://arxiv.org/abs/2202.01725) to approximate PI for rapid calculation.

To train RipsNet with generated PCD and PIs, run the following code:

```bash
python train_ripsnet.py
```

### Training student network

Finally, to mimic PIs of the teacher, train student network with *topology distillation loss*, KD loss, and CE loss.

To train student network, 

```bash
sh train_student.sh
```

## Evaluation

### Checkpoints

We included the weights and PI of some teacher networks, WRN-40-2 and ResNet56, in “save_t_models” and “ripsnet” folders, respectively. The remaining teacher networks will be uploaded later.

## Comparison

### Results on CIFAR-100:

Homogeneous architectures

![스크린샷 2024-05-22 오전 11.25.15.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/8e15960d-1fd8-414c-95d2-2a79c370cea8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.25.15.png)

Heterogeneous architectures

![스크린샷 2024-05-22 오전 11.26.13.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/6fd3f667-de88-4c8f-abed-cd352696bd90/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.26.13.png)

### Results on ImageNet-1K:

Homogeneous architectures

![스크린샷 2024-05-22 오전 11.26.23.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/ea012321-93eb-4427-a35f-5d903c2fd022/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.26.23.png)

Heterogeneous architectures

![스크린샷 2024-05-22 오전 11.26.35.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/4c0d6e48-ef59-4ffa-bee7-c9d90616f781/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.26.35.png)

## Visualization

### Exact and approximated PIs

**Teacher: VGG13, student: MobileNetV2**

![스크린샷 2024-05-22 오전 11.28.01.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/81526018-9d5a-471e-8b3d-f5a8dcd98925/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.28.01.png)

### UMAP

**CIFAR-100:**

![스크린샷 2024-05-22 오전 11.28.19.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/000722fa-64b3-42e8-b73e-280be5fafebc/ac6b4015-246d-40f6-bdf6-ff73cecdd114/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-22_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.28.19.png)