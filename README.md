# Advbench

Code and data of the EMNLP 2022 paper **"Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP"**[[PDF](https://arxiv.org/pdf/2210.10683v1.pdf)] .

## Overview

In this paper, we rethink the research paradigm of textual adversarial samples in security scenarios.
We discuss the deficiencies in previous work and propose our suggestions that the research on the **S**ecurity-**o**riented **ad**versarial **NLP (SoadNLP) should:**
(1) evaluate their methods on security tasks to demonstrate the real-world concerns;
(2) consider real-world attackers' goals, instead of developing impractical methods. 
To this end, we first collect, process, and release a security datasets collection **advbench**. Then, we reformalize the task and adjust the emphasis on different goals in SoadNLP. Next, we propose a simple method based on heuristic rules that can easily fulfill the actual adversarial goals to simulate real-world attack methods.We conduct experiments on both the attack and the defense sides on Advbenchmark. 
Experimental results show that our method has higher practical value, indicating that the research paradigm in SoadNLP may start from our new benchmark.

<img src="figs/main.png" alt="main" style="zoom:50%;" />

## Dependencies

```
pip install -r requirements.txt
```

Maybe you need to change the version of some libraries depending on your servers.


## Data Preparation

First, you need to create the file `data` to store dataset:

```
cd Advbench
mkdir data
```

Then you need to download the data from Google Drive[[data](https://drive.google.com/drive/folders/1_2q2282ZEoE_iPg8Q4ILGeB_aAkcP43v?usp=sharing)] .

We provide the original dataset (**ori_dataset**), the processed dataset (**rel_dataset)**  the experimental dataset (**exp_dataset**) and a pure compression package for experiments. If you just want to reproduce the experiment, you shold download the **data.zip** and save it into `/data`, then unpakage the zip file with the following command:
```
unzip data.zip
```

If you want to use our benchmark for further research, please download **rel_dataset**. If you want to use raw dataset to process the data yourself, you can download **ori_dataset**. **Exp_dataset** is just an uncompressed format of **data.zip** .

## Experiments

First, you need to create the file `model` and `output` to respectively store fine-tuned model and adversarial output dataset.
```
mkdir model
mkdir output
```

Then you should fine-tune the pre-trained model on our security datasets collection **Advbench**.

```
bash scripts/train.sh
```

To conduct the baseline attack experiments in our settings:

```
bash scripts/base_attack.sh
```

To conduct attack experiments via ROCKET in our settings:

```
bash scripts/rocket.sh
```

## Citation
Please kindly cite our paper:

```
@article{chen2022should,
  title={Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP},
  author={Chen, Yangyi and Gao, Hongcheng and Cui, Ganqu and Qi, Fanchao and Huang, Longtao and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2210.10683},
  year={2022}
}
```

