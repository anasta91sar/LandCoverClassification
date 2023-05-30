# LandCoverClassification

This repository contains tools and functions developed for my thesis entitled Land cover classification using machine-learning techniques applied to fused multi-modal satellite imagery and time series data.

The objective of this project is to classify an area in ‘Artificial’, ‘Bare Soil’, ‘Cropland’, ‘Dense Forest’, ‘Grassland’, ‘Low-density Urban’, ‘Low/Sparse Vegetation, and ‘Water’ classes with given training polygon samples using kNN and Random Forests Machine Learning algorithms. The study addresses several questions, including the impact of thermal information, elevation, and topography on the classification accuracy, as well as the utilization of time series data to enhance the results compared to using only the multispectral information as input.

To do so, Landsat-8 and Landsat-9 multi-spectral and thermal imagery are employed, as well as topographic data of the area of interest. Additionally, ASTER GDEM data was used for elevation information and the generation of two derivatives: the aspect and the slope of the study area. These factors, along with their temporal variability (time series), are considered crucial as the spectral properties of certain key classes (specifically those related to vegetation and agricultural activities) are influenced by the phenological cycle.

The order of the scripts is as indicated by their name (X in 0X_name.py).

For further information please see my thesis manuscript here: (to be added when published)

Anastasia Sarelli (2023). Land cover classification using machine-learning techniques applied to fused multi-modal satellite imagery and time series data.
Master degree thesis, 30/ credits in Master in Geographical Information Science
Department of Physical Geography and Ecosystem Science, Lund University
