# Segmenting Glomeruli Functional Tissue Units in Human Kidneys

## Project Overview
This project focuses on segmenting glomeruli functional tissue units (FTUs) in human kidney images as part of the "Hacking the Kidney" program within the Human BioMolecular Atlas Program (HuBMAP). The goal is to develop effective methods for identifying and characterizing FTUs across various tissue samples.

## Key Features
1. Implementation of U-Net architecture for image segmentation
2. Data augmentation techniques to address overfitting
3. Exploration of Segment Anything Model (SAM) for zero-shot segmentation
4. Development of a novel segmentation + classification pipeline

## Methodology
- U-Net based segmentation with enhancements:
  - Deep Fourier Upsampling
  - Data augmentation (image flipping, training mixup)
  - Normalization techniques
- SAM and Fast-SAM exploration:
  - Zero-shot segmentation capabilities
  - Optimization for computational efficiency
- Classification approach:
  - Conversion of segmentation task to classification
  - Utilization of Fast-SAM with text prompts

## Dataset
The project uses high-resolution TIFF images of kidney tissue, processed into 512x512 pixel tiles. Data preprocessing includes:
- Removal of invalid tiles
- Normalization based on valid tile statistics

## Results
- Successful segmentation of glomeruli with U-Net
- Improved performance through data augmentation and mixup techniques
- Exploration of SAM's potential in medical image segmentation
- Development of a computationally efficient pipeline using Fast-SAM and classification

## Tools and Libraries
- PyTorch
- Albumentations
- Segment Anything Model (SAM)
- Fast-SAM

## Authors
- Xiao Lingao
- Dong Luojie
- Sirin Nadir Kaan
- Jose Antonio Franca
- Heng Wei Jie

## Acknowledgments
This project is part of the coursework for Machine Learning at Nanyang Technological University, School of Computer Science and Engineering.
