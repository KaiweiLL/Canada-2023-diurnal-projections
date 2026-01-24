# Canada-2023-diurnal-projections

# Future Normalization of North American Fire Extremes

This repository contains the code supporting the paper "Was Canada's 2023 fire season a preview of things to come in North America?" by Kaiwei Luo, Xianli Wang, Dante Castellanos-Acuna, and Mike Flannigan.

## Overview

This research quantifies how North America's diurnal burning windows—specifically Active Burning Day potential (ABDp) and Overnight Burning Event potential (OBEp)—will evolve under climate change. 
By linking satellite observations (GOES-R) with CMIP6 climate models via machine learning, we assess whether recent catastrophic fire seasons (Canada 2023 and Western U.S. 2020/2021) are statistical outliers or the "new normal"



## Workflow and Code Organization

### 1. Data Processing and Algorithms

This section includes the initial data preparation and processing of GOES satellite data:

- **01_Download.R**: Downloads GOES-16, GOES-17, and GOES-18 active fire products.
- **02_Processing.R**: Processes raw GOES active fire detection data.
- **03_Projecting.R**: Projects the processed data to consistent geographic coordinates.
- **04_Events_extraction_2017_2023.R**: Identifies hourly fire diurnal cycles and extracts active burning days (ABD) and overnight burning events (OBE) in North America, 2017-2023.
- **05_NA_daily_extract_fire weather_2017_2023.R**: Extracts and calculates daily fire weather metrics.
- **06_Events and fire weather match.R**: Match the ABD and OBE with daily fire weather.

### 2. ML model training

Trains a hierarchical Random Forest model to link fire weather to ABD and OBE

- **01_ML_training.py**: link fire weather to ABD and OBE.

### 3. Future climate and fire weather projections 

This section details the downscaling and projection of fire weather under different climate scenarios, 4 CMIP6 models, 4 SSPs, 2 periods.
four CMIP6 models (CanESM, UKESM, EC-Earth, GFDL), four SSPs (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5), mid-century (2041-2070) and late-century (2071-2100); in total 32 sets of future daily meteorological variables, and thus 32 sets of future daily fire weather variables.
Please contact us for the codes and datasets for future climate or fire weather projections.

- **01_Future_ABDp_OBEp_projction.py**: Applies trained ML models to future fire weather projections to estimate ABDp and OBEp for mid-century (2041-2070) and late-century (2071-2100) under variaous climate model and SSPs combination.

### 4. Analysis and Benchmarking

Scripts for spatial analysis, latitudinal amplification, and benchmarking recent extremes:

- **01_Spatial_Analysis.py**: Continental analysis to reveal magnitude and spatial pattern of changes in ABDp and OBEp, e.g, multi-model agreement and identifies areas of "unanimous" fire potential increase.
- **02_Latitudinal_Gradient_Analysis.py**: Analyzes the amplification of fire risk across boreal, temperate, and subtropical biomes.
- **03_Canada_2023_Normalization.py**: Benchmarks the 2023 Canadian fire season against future distributions to determine its frequency of occurrence
- **04_Western_US_Comparison.py**: Evaluates the 2020/2021 Western U.S. extremes within mid- and late-century climate envelopes


## Key Findings

- **Overnight Burning Escalation**: OBEp increases outpace ABDp in both magnitude and spatial coherence, signaling a systematic weakening of the diurnal firebreak.
- **Boreal Amplification**: High-latitude regions show the strongest response, with boreal overnight potential doubling to tripling by late-century under high-warming scenarios.
- **The "New Normal"**: Conditions underlying Canada's record-breaking 2023 season become routine by mid-century, even under ambitious mitigation (SSP1-2.6); whereas Western U.S.2020/2021 extremes approach typical conditions only by late-century and under higher warming scenarios
- **Spatial Synchronization**: Future fire risk shifts from localized hotspots to continent-wide, synchronized extremes, challenging current resource-sharing strategies.

## Data Requirements

The analysis requires several datasets:

- GOES active fire images (nc) from Amazon Web Service S3 Explorer
- Fire perimeters (shapefile) from NBAC, MTBS, and CWFP
- ERA5-based daily and hourly fire weather (nc), 1990-2023
- Future climate and fire weather projections (nc), 32 sets from  4 CMIP6 models, 4 SSPs, 2 periods
- Biome categorization (shapefile)

## Software Requirements

### R Dependencies (ABH Analysis and Data Processing)
- rgdal, raster, sp
- dplyr, tidyr, reshape2
- ggplot2, ggsci
- lutz, suncalc
- foreach, doParallel, parallel, tcltk, doSNOW
- caret, MKinfer, pROC

### Python Dependencies (PHB Analysis)
- xarray, numpy, pandas
- matplotlib, cartopy
- geopandas, regionmask
- pymannkendall
- scipy
- scikit-learn

## Note

The datasets required for the code are listed in the data requirements and are all publicly available, except for future climate and fire weather projections (datasets coming soon)

Please be aware that this code was not crafted by a professional software developer, hence it might not exhibit the highest level of elegance or efficiency. This code has been processed with the assistance of a large language model for clarity and organization when uploaded to GitHub. Should you have any feedback or recommendations concerning the code, we encourage you to share them with us. Additionally, if you have any inquiries about the code, the data, or the analysis overall, please don't hesitate to reach out.

## Citation

If you use this code or data in your research, please cite: Luo, K., Wang, X., Castellanos-Acuna, D., & Flannigan, M. (2025). Was Canada's 2023 fire season a preview of things to come in North America?
