# Do Topological Characteristics Help in Knowledge Distillation?
## Accepted in ICML 2024

[[Paper]](💥)

![architecture (1)](https://github.com/jekim5418/TopKD/assets/60121575/97a120bd-9245-4c41-9d0b-d04f9d4980ed)

> Do Topological Characteristics Help in Knowledge Distillation?
>
>International Conference on Machine Learning (ICML) 2024\\
>Jungeun Kim*,  Junwon You*, Dongjin Lee*, Ha Young Kim, Jae-Hun Jung\\
>Yonsei University & POSTECH
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

<img width="771" alt="스크린샷 2024-05-22 오전 11 25 15" src="https://github.com/jekim5418/TopKD/assets/60121575/5b26ac5d-40ae-42a8-9e86-dcd74a22b840">

Heterogeneous architectures

<img width="783" alt="스크린샷 2024-05-22 오전 11 26 13" src="https://github.com/jekim5418/TopKD/assets/60121575/81f30edc-c61c-42ac-a635-9c2d542da157">

### Results on ImageNet-1K:

Homogeneous architectures

<img width="737" alt="스크린샷 2024-05-22 오전 11 26 23" src="https://github.com/jekim5418/TopKD/assets/60121575/2eb1610a-b6b6-49ba-8949-a3be4cd5041e">

Heterogeneous architectures

<img width="676" alt="스크린샷 2024-05-22 오전 11 26 35" src="https://github.com/jekim5418/TopKD/assets/60121575/cfca7415-812b-4b64-be69-22001f934efb">

## Visualization

### Exact and approximated PIs

**Teacher: VGG13, student: MobileNetV2**

<img width="550" alt="스크린샷 2024-05-22 오전 11 28 01" src="https://github.com/jekim5418/TopKD/assets/60121575/96e3c404-2e69-4ca8-ae43-4c98e3689bf3">

### UMAP

**CIFAR-100:**

<img width="689" alt="스크린샷 2024-05-22 오전 11 28 19" src="https://github.com/jekim5418/TopKD/assets/60121575/da2ede80-1380-4818-b8c2-a3b5d577a069">

**ImageNet-1K**

<img width="673" alt="스크린샷 2024-05-22 오후 2 14 11" src="https://github.com/jekim5418/TopKD/assets/60121575/a78897d0-d744-48a6-bc4b-870d8a895653">
