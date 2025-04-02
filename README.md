# Robust Spatial Multi-Omics Integration with SpaBalance via Coordinated Gradient Learning and Feature Decoupling
This repository contains the necessary SpaBalance scripts to reproduce the benchmark results presented in the paper. We also provide experimental data, which can be found in the data folder. All experiments can be reproduced using the provided Tutorial.py script. 

![](https://github.com/nudt-bioinfo/SpaBalance/blob/main/framework.png)

## Overview
Integrating multiple data modalities based on spatial information remains an unmet need in the analysis of spatial multi-omics data. Here, we propose SpaBalance, a spatial multi-omics deep integration model designed to decode spatial domains by leveraging graph neural networks (GNNs) and multilayer perceptrons (MLPs). SpaBalance first performs intra-omics integration by capturing within-modality relationships using spatial positioning and omics measurements, followed by cross-omics integration through a multi-head attention mechanism. To effectively learn shared features while preserving key modality-specific information, we employ a dual-learning strategy that combines modality-specific private learning with cross-modality shared learning. Additionally, to prevent any single modality from dominating the integration process, we introduce a multi-modal balance learning approach. We demonstrate that SpaBalance achieves more accurate spatial domain resolution across diverse tissue types and technological platforms, providing valuable biological insights into cross-modality spatial correlations.

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* scikit-misc==0.2.0
* seaborn==0.13.2 
* torch-geometric==2.6.1
* tqdm==4.64.0
* matplotlib==3.4.2
* R==4.0.3

## Tutorial
For a detailed step-by-step tutorial, please refer to the Tutorial.py script.

## Data
We tested simulated dataset and experimental datasets, including:  

- **mouse spleen dataset** obtained using the SPOTS technique (Ben-Chetrit et al., 2023).  
- **mouse thymus dataset** generated using Stereo-CITE-seq technology (unpublished).  
- **mouse brain spatial epigenome-transcriptome dataset** (Zhang et al., 2023).  
- **human lymph node dataset** obtained using the 10x Visium CytAssist technology (Long et al., 2024).  
- **simulated dataset** containing three omics (Long et al., 2024).

The SPOTS data for the mouse spleen is available in the **GEO database** (accession number **GSE198353**, [GEO Accession Viewer](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353)). The Stereo-CITE-seq data for the mouse thymus is provided by **BGI**. The spatial epigenome-transcriptome data for the mouse brain is available on **AtlasXplore** ([AtlasXplore](https://web.atlasxomics.com/visualization/Fan)).  The human lymph node dataset and simulated dataset can be accessed at the [GitHub repository of JinmiaoChenLab/SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue).

