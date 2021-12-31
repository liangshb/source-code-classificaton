<div align="center">

# Source Code Classification

</div>

## Description

Source code classification

## Preprocess

## Datasets

1. [SySeVR](https://github.com/SySeVR/SySeVR): 8:1:1 and 8:2
2. [muVulDeePecker](https://github.com/muVulDeePecker/muVulDeePecker): 8:1:1 and 8:2
3. [VulDeePecker](https://github.com/CGCL-codes/VulDeePecker): 8:1:1 and 8:2
4. [DeepWukong](https://github.com/DeepWukong/Dataset): 8:1:1
5. [BufferOverrun](https://github.com/mjc92/buffer_overrun_memory_networks): pre-defined
6. [D2A](https://github.com/IBM/D2A): pre-defined
7. [MSR](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset): 8:1:1
8. [GHPR](https://github.com/feiwww/GHPR_dataset): 8:1:1
9. [ReVeal](https://github.com/VulDetProject/ReVeal): 8:1:1
10. [CodeXGLUE](https://github.com/microsoft/CodeXGLUE): pre-defined

## Baselines

1. SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities
2. VulDeePecker: A Deep Learning-Based System for Vulnerability Detection
3. $\mu$VulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection
4. FTCLNet: Convolutional LSTM with Fourier Transform for Vulnerability Detection
5. An Automatic Source Code Vulnerability Detection Approach Based on KELM
6. DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network: DeepWukong
7. End-to-end prediction of buffer overruns from raw source code via neural memory networks: BufferOverrun
8. Vulnerability Detection with Fine-grained Interpretations: MSR and Reval

## Models

1. CNN(without or with tag)
2. CNN with highway(without or with tag)
3. Bidirectional GRU
4. Bidirectional LSTM
5. Stacked Alternating LSTM
6. Stacked Bidirectional LSTM

## Configurations

1.  token_embedding_size: 50, 100, 150
2.  type_embedding_size: 50

## Innovation Points

1. Model
   1. use cnn with highway
   2. use tag embedding
   3. interpret result
   4. compare symbolize or not
   5. maybe
      1. CRF: Conditional Random Field
      2. pre-trained embedding: FastText vs. Word2Vec

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/liangshb/source-code-classificaton sccls
cd sccls

# [OPTIONAL] create conda environment
conda create -n sccls python=3.8
conda activate sccls

# install pytorch according to instructions
# https://pytorch.org/get-started/
# example: conda install pytorch cudatoolkit=11.3 -c pytorch

# install requirements
pip install -r requirements.txt
```
