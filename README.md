# nERdy

## network analysis of endoplasmic reticulum dynamics with image processing and deep learning methods
Ashwin Samudre, Guang Gao, Ben Cardoen, Ivan Robert Nabi, Ghassan Hamarneh

Pre-print: [https://www.biorxiv.org/content/10.1101/2022.05.17.492189v1](https://www.biorxiv.org/content/10.1101/2024.02.20.581259v1.abstract)

## Abstract
The endoplasmic reticulum (ER) comprises smooth tubules, ribosome-studded sheets, and peripheral sheets that can present as tubular matrices. ER shaping proteins determine ER morphology, however, their role in tubular matrix formation requires reconstructing the dynamic, convoluted ER network. Existing reconstruction methods are sensitive to parameters or require extensive annotation and training for deep learning. We introduce nERdy, an image processing based approach, and nERdy+, a D4-equivariant neural network, for accurate extraction and representation of ER networks and junction dynamics, outperforming previous methods. Comparison of stable and dynamic representations of the extracted ER structure reports on tripartite junction movement and distinguishes tubular matrices from peripheral ER networks. Analysis of live cell confocal and STED time series data shows that Atlastin and Reticulon 4 promote dynamic tubular matrix formation and enhance junction dynamics, identifying novel roles for these ER shaping proteins in regulating ER structure and dynamics.

## Repository structure
nERdy: Image processing method

nERdy+: D4-equivariant segmentation method

analysis: scripts for junction analysis pipeline and to generate all the results in the paper

## Cite
If you find this article useful, please cite
```
@article{samudre2024nerdy,
  title={nERdy: network analysis of endoplasmic reticulum dynamics},
  author={Samudre, Ashwin and Gao, Guang and Cardoen, Ben and Nabi, Ivan Robert and Hamarneh, Ghassan},
  journal={bioRxiv},
  pages={2024--02},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
