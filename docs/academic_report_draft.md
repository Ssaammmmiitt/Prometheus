  
Kathmandu University

Department of Computer Science and Engineering

Dhulikhel, Kavre

A Project Report

on

“Prometheus: A Wildfire Prediction System”

\[Code No.: COMP 308\]

(For partial fulfillment of Year III / Semester II in Computer Engineering)

Submitted by

Baldeep Karki (03230922)

Sammit Poudyal (03232922)

Submitted to

Mr. Suman Shrestha

Department of Computer Science and Engineering

January 12, 2026

Bona fide Certificate

This project work on

“Prometheus: A Wildfire Prediction System”

is the bona fide work of

“

Baldeep Karki (03230922),

Sammit Poudyal (03232922)

”

who carried out the project work under my supervision.

Project Supervisor

Dr. Rabindra Bista

DOCSE

Acknowledgements

We would like to thank everyone who helped make this project possible. First,we’re grateful to the Department of Computer Science and Engineering, Kathmandu Uni-versity, for giving us the opportunity to work on Prometheus: A Wildfire Prediction System

We owe special thanks to our supervisor, Dr. Rabindra Bista, whose guidance made all the difference. His feedback helped us refine our approach to the problem, and his encouragement kept us going through the challenges we faced. We learned a great deal from his insights and expertise. This project wouldn’t have been possible without several open tools and datasets. We’re particularly thankful to:

* Google Earth Engine (GEE) team for providing access to satellite im- agery and geospatial data that formed the backbone of our system.

* NASA FIRMS team for providing access to satellite wildfire images and geospatial data for wildfire data.

* Kaagle and various machine learning communities where we found helpful resources when we encountered challenges.

ii

Abstract

Wildfires pose a significant threat to ecosystems, human settlements, and air quality, particularly in regions with complex terrain and seasonal climate variabil-ity. Accurate early prediction of wildfire occurrence is critical for effective disaster preparedness and mitigation. This study presents a wildfire prediction system that leverages spatiotemporal satellite data and deep learning techniques to forecast fire occurrence in advance. The proposed system utilizes multi-source remote sensing inputs, including vegetation indices, land surface temperature, and historical fire data, modeled using a Convolutional Long Short-Term Memory (ConvLSTM) net-work to capture both spatial patterns and temporal dynamics. The model is trained on time-sequenced satellite observations and evaluated using classification metrics such as precision, recall, and F1-score. Experimental results demonstrate that the system is capable of learning meaningful fire-prone patterns over time and provides early warning signals for potential wildfire events. This approach highlights the po-tential of deep learning–based spatiotemporal models for supporting wildfire risk assessment and decision-making in disaster management.

Keywords: CONVLSTM, precision, recall, F1-score, spatial, temporal, spatiotem-poral, disaster management

iii

Contents

| 1 | Introduction |  |  | 1 |
| ----- | :---- | :---- | ----- | ----: |
|  | 1.1 | Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  | 1 |
|  | 1.2 | Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  | 1 |
|  | 1.3 | Motivation and Significance  . . . . . . . . . . . . . . . . . . . . . |  | 2 |
| 2 | Related Works |  |  | 3 |
|  | 2.1 | Spatial Statistical Models for Wildfire Prediction  . . . . . . . . . . |  | 3 |
|  | 2.2 | CNN-based Spatial Models for Wildfire Susceptibility  . . . . . . . |  | 3 |
|  | 2.3 | Spatiotemporal Deep Learning for Daily Fire Forecasting . . . . . . |  | 4 |
|  | 2.4 | Review of Deep Learning Advances in Wildfire Prediction  . . . . . |  | 4 |
|  | 2.5 | Operational System and Future Directions . . . . . . . . . . . . . . |  | 4 |
| 3 | Design and Implementation |  |  | 5 |
|  | 3.1 | System Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . |  | 5 |
|  |  | 3.1.1 | Objective  . . . . . . . . . . . . . . . . . . . . . . . . . . . | 5 |
|  | 3.2 | System Requirement Specifications  . . . . . . . . . . . . . . . . . |  | 5 |
|  |  | 3.2.1 | Software Requirements . . . . . . . . . . . . . . . . . . . . | 5 |
|  |  | 3.2.2 | Hardware Requirements  . . . . . . . . . . . . . . . . . . . | 7 |
|  | 3.3 | System Design  . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  | 8 |
|  |  | 3.3.1 | Architectural Design  . . . . . . . . . . . . . . . . . . . . . | 8 |
|  |  | 3.3.2 | Module or Component Design . . . . . . . . . . . . . . . . | 8 |
|  |  | 3.3.3 | Data Design  . . . . . . . . . . . . . . . . . . . . . . . . . | 8 |
|  | 3.4 | Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . |  | 10 |
|  |  | 3.4.1 | Software Implementation . . . . . . . . . . . . . . . . . . . | 10 |
|  |  | 3.4.2 | Machine Learning Implementation  . . . . . . . . . . . . . | 10 |
|  | 3.5 | Algorithms and Flowcharts . . . . . . . . . . . . . . . . . . . . . . |  | 11 |
|  | 3.6 | Tools and Technologies Used . . . . . . . . . . . . . . . . . . . . . |  | 14 |
|  |  | 3.6.1 | Summary of Required Content by Project Type  . . . . . . . | 15 |
| 4 | Results and Discussion |  |  | 16 |
|  | 4.1 | Implemented Features . . . . . . . . . . . . . . . . . . . . . . . . . |  | 16 |
|  | 4.2 | Results and Performance Analysis  . . . . . . . . . . . . . . . . . . |  | 16 |
|  |  | 4.2.1 | Wildfire Prediction System . . . . . . . . . . . . . . . . . . | 16 |
|  |  | 4.2.2 | Evaluation Metrics  . . . . . . . . . . . . . . . . . . . . . . | 17 |
|  |  | 4.2.3 | Precision Recall Curve . . . . . . . . . . . . . . . . . . . . | 17 |
|  |  | 4.2.4 | Precision Recall Curve . . . . . . . . . . . . . . . . . . . . | 18 |

iv

| 4.3 | Challenges and Limitations . . . . . . . . . . . . . . . . . . . . . . | 19 |
| :---- | :---- | ----: |
| 4.4 | Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 19 |
| 5  Conclusion and Future Works |  | 20 |
| 5.1 | Limitations  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 20 |
| 5.2 | Future Enhancements . . . . . . . . . . . . . . . . . . . . . . . . . | 21 |
| References |  | 22 |
| Appendix-A |  | 22 |
| A.1 | Additional Figures and Screenshots  . . . . . . . . . . . . . . . . . | 23 |
| A.2 | Extended Tables and Results  . . . . . . . . . . . . . . . . . . . . . | 23 |
| A.3 | Algorithms and Pseudocode  . . . . . . . . . . . . . . . . . . . . . | 24 |
| A.4 | Source Code Snippets (Optional) . . . . . . . . . . . . . . . . . . . | 24 |
| A.5 | User Manuals or Installation Guides (Optional)  . . . . . . . . . . . | 24 |
| A.6 | Ethical Approval or Consent Forms (If Applicable)  . . . . . . . . . | 24 |
| A.7 | Additional Documentation  . . . . . . . . . . . . . . . . . . . . . . | 24 |

v

List of Figures

3.1	Flowchart of the ConvLSTM-based wildfire prediction pipeline . . .	13

4.1	Flowchart of the ConvLSTM-based wildfire prediction pipeline . . .	17

vi

List of Tables

| 3.1 | Hyperparameter Comparison of Wildfire Prediction Models . . . . . | 11 |
| ----: | :---- | ----: |
| 3.2 | Summary of Design and Implementation Content for Current Project | 15 |

vii

viii

Chapter 1

Introduction

1.1	Background

In a country like Nepal with over 46% forest occupancy, there remains a huge risk of wildfire. Wildfires have been one of the major causes of damage to forest area, human lives and properties in Nepal averaging about 3000 wildfire incidents per year \[1\], according to ICIMOD. In recent years, the frequency of wildfires have increased significantly due to global warming, prolonged dry season and changing vegetation patterns. These incidents cause Nepal’s cities to choke under toxic air every year.

Although satellite based fire monitoring systems exist, like the Forest Fire De-tection and Monitoring System in Nepal under the Servir Hindu Kush Initiative by ICIMOD, FIRMS by NASA, they often detect fire after ignition. This highlights the need for predictive modeling using climate, topography ,vegetation and so on. The predictive models made for wildfire prediction in Nepal use machine learning algorithms like random forest which rely on discrete data.

Recent developments in deep learning architectures, such as Convolutional Long Short-Term Memory (ConvLSTM) networks for spatio-temporal modeling have demonstrated strong potential in wildfire prediction. At the same time, cloud-based platforms like Google Earth Engine (GEE) facilitate the efficient processing and management of large geospatial datasets. Despite these advances, many existing wildfire prediction systems remain technically complex and inaccessible to general users, highlighting the need for an integrated solution that combines reliable wildfire analysis.

1.2	Objectives

The objective of this project were:

* To integrate heterogeneous data sources, including satellite imagery, NDVI, terrain data into a unified data analysis pipeline.

* To predict potential flood risks by leveraging environmental, climatic, and topographic information for early warning and preparedness.

1

1.3	Motivation and Significance

The motivation behind this project arises from the numerous wildfire inci-dents that occur in Nepal, every year causing loss of life, loss of home, loss of biodiversity and yearly increase in air pollution, most notably the fire in Jan 2021 which burned across 22 out of 77 adminsitrative districts of Nepal. This demonstrates the lack of wildfire prediction systems in Nepal.

This project was motivated by the need to provide actionable flood informa-tion in a country with limited wildfire monitoring infrastructure. Given the scope of this academic project, it focused on leveraging accessible satellite imagery, wildfire data, and predictive modeling techniques to create a system capable of forecasting wildfire risks at the local level.

The significance of this work lies not in preventing wildfire directly, but in en-abling timely awareness and preparedness. Even approximate or probabilistic predictions can provide crucial lead time for response planning, helping to save lives, protect critical infrastructure, and reduce the social and economic impacts of wildfire-related disasters in Nepal.

2

Chapter 2

Related Works

2.1	Spatial Statistical Models for Wildfire Prediction

Early wildfire prediction studies primarily relied on statistical and regression-based approaches to model fire occurrence using environmental and human-related variables. Li et al. Li et al. (2021) developed a Geographically Weighted Logistic Regression (GWLR) model to predict forest fires in Yunnan Province, China, using 23 explanatory variables derived from NASA VIIRS active fire data at a spatial resolution of 375 m. By allowing regression coefficients to vary spatially, the GWLR model captured regional heterogeneity more effec-tively than traditional logistic regression models.

The main strength of this approach lies in its interpretability and ability to model spatial non-stationarity. The model achieved an Area Under the Curve (AUC) of 0.92 and an accuracy of 82.7%, demonstrating improved perfor-mance over global regression techniques. However, GWLR depends heavily on manually selected features and assumes static relationships over time. It does not explicitly model temporal evolution, limiting its effectiveness for dy-namic wildfire processes. In comparison, the Prometheus system moves be-yond static spatial modeling by incorporating temporal dependencies through deep learning.

2.2	CNN-based Spatial Models for Wildfire Susceptibility

Zhang et al. Zhang et al. (2019) applied deep learning using Convolutional Neural Networks (CNNs) to model forest fire susceptibility in Yunnan Province. Their approach used 14 key environmental variables and represented each data sample as a 25×25 pixel patch, where each pixel contained multiple features such as vegetation, elevation, and land cover. CNNs automatically learned spatial patterns, improving prediction accuracy without extensive fea-ture engineering.

The CNN model achieved 95.81% accuracy, illustrating the power of deep learning to capture spatial complexity. Nevertheless, it primarily focused on static spatial features and did not incorporate temporal dynamics. Prometheus advances beyond this limitation by combining CNN-style spatial learning with temporal modeling using ConvLSTM.

3

2.3	Spatiotemporal Deep Learning for Daily Fire Forecasting

Prapas et al. Prapas et al. (2023) developed a deep learning framework for daily wildfire danger forecasting across Greece. They constructed a large spatiotemporal datacube covering 2009–2020 by integrating data from ERA-5 Land, MODIS, and Copernicus datasets. Several models were trained, in-cluding Random Forest, CNN, RNN, and ConvLSTM.

Their experiments demonstrated that models capable of learning both spa-tial and temporal patterns, particularly ConvLSTM, outperformed traditional indices and purely spatial models. This work highlighted the importance of modeling dynamic environmental conditions for realistic wildfire prediction.

2.4 Review of Deep Learning Advances in Wildfire Predic-tion

Xu et al. Xu et al. (2025) reviewed recent deep learning methods for wildfire prediction, emphasizing effective integration of remote sensing and environ-mental data. They classified approaches into three categories: time-series forecasting, image segmentation, and spatiotemporal prediction. Challenges identified included data scarcity and limited generalization across regions. ConvLSTM and similar architectures were highlighted as balancing accuracy and interpretability, making them suitable for dynamic events like wildfires.

2.5	Operational System and Future Directions

Overall, wildfire prediction research has evolved from static statistical mod-els, such as GWLR, to spatiotemporal deep learning methods. Early studies identified key environmental variables but ignored temporal dynamics, while CNNs improved spatial feature learning without modeling time. ConvLSTM approaches integrate both spatial and temporal dependencies, offering more accurate and adaptable predictions.

The Prometheus system leverages multi-source spatial and temporal data in a ConvLSTM framework for patch-level wildfire prediction. This can be further refined to pixel-wise segmentation, enabling precise mapping of fire-prone ar-eas, better resource allocation, and early warning. Pixel-level predictions cap-ture intra-patch variability, providing higher spatial resolution and actionable insights for wildfire management.

4

Chapter 3

Design and Implementation

The model developed is a

3.1	System Overview

This section provides a high-level description of the Prometheus wildfire pre-diction system, explaining its objective, major components, and interactions between them.

3.1.1	Objective

The primary objective of Prometheus is to provide accurate, regionally rele-vant, and temporally aware predictions of wildfire occurrences. By leverag-ing multi-source spatial and temporal data, the system aims to support early warning, resource allocation, and strategic planning for wildfire management.

3.2	System Requirement Specifications

3.2.1	Software Requirements

The implementation of the flood detection and prediction system relies on the following software stack:

– Programming Languages:

* Python (3.10+): Used for backend logic, data preprocessing, and deep learning model implementation.

  * JavaScript (ES6+): Used for frontend interactivity, API calls and fetching data from Google Earth Engine.

    – Frameworks and Libraries:

  * PyTorch: The primary deep learning framework for training the ConvLSTM model.

5

* Leaflet.js / Mapbox: Used for rendering interactive maps and han-dling geospatial data on the client side.

  * Rasterio: Used for reading and manipulating Sentinel-1 ‘.tif’ satel-lite imagery.

  * NumPy: Used for matrix operations and handling time-series datasets.

    – Development Tools and Environments:

  * Visual Studio Code (VS Code): The primary Integrated Develop-ment Environment (IDE).

  * Kaggle Kernels / Google Colab: Cloud-based environments pro-viding NVIDIA T4 GPUs for model training.

  * Git & GitHub: Used for version control and source code manage-ment.

    – Operating System Requirements:

  * Development: Windows 10/11 or Linux (Ubuntu 20.04 LTS).

  * Deployment: A Linux-based server environment.

  Functional Requirements

* User Input & Interaction:

  The web interface shall allow users to select a Region of Interest (ROI) on a map via clicking or drawing a bounding box.

* Data Acquisition & Preprocessing:

  The system shall automatically retrieve and preprocess 16-day cumu-lative multi-source satellite imagery (MODIS, VIIRS) for the selected ROI.

* Wildfire Prediction Engine:

  The system shall execute the trained ConvLSTM model to process tem-poral sequences of spatial patches and predict wildfire occurrence.

* Visualization & Reporting:

  The system shall display predicted fire patches on a geospatial map with color-coded risk levels.

  Non-Functional Requirements

* Latency:

6

The model generates a fire prediciton within seconds of being fed the image arrays.

* Accuracy:

  The Prediction Model acheives an AUC score of 0.718. This can further be enhanced by reducing the dataset to fire-prone areas only, increas-ing the relative number of positive samples. The validation loss for the model is 0.157.

* Reliability: The model shall handle missing or invalid satellite data (e.g., values marked as \-9999 or no data due to orbit gaps) gracefully, without terminating unexpectedly.

  3.2.2	Hardware Requirements

  Since this is a machine learning-based project involving computationally in-tensive deep learning models, specific hardware configurations are required for both the development (training) and deployment phases.

  – Training Environment (Development):

  * GPU: An NVIDIA GPU (e.g., Tesla T4, P100, or RTX 30-series) with at least 16 GB of VRAM is required to accelerate model train-ing and support large batch sizes.

  * RAM: A minimum of 8 GB system memory is required, with 16 GB recommended to efficiently handle data loading and prepro-cessing pipelines.

  * Storage: At least 10 GB of available SSD storage is required to store the Sen1Floods11 Essentials dataset (hand-labeled image chips) and saved model checkpoints.

  * Processor: A multi-core CPU (Intel i7 or equivalent) is recom-mended to support efficient data loading, preprocessing, and aug-mentation operations.

    – Deployment Environment (Server/User):

  * Server Side: A standard cloud instance (e.g., AWS EC2 or Google Compute Engine) with basic GPU support or a high-performance CPU is sufficient for running model inference through the FastAPI backend.

  * Client Side: Any standard laptop or desktop device equipped with a modern web browser and a stable internet connection is sufficient to access and use the web-based application.

7

3.3	System Design

This section explains how the system was designed prior to implementation.

3.3.1	Architectural Design

3.3.2	Module or Component Design

* Data Processing Module: Performs preprocessing, normalization, and fea- ture extraction for both satellite imagery and time-series data.

* Wildfire Prediction Module: Predicts wildfire occurrence probability for a given location using historical, geographical, climatic features.

  3.3.3	Data Design

  The system relies on multi-temporal raster datasets to predict wildfire occur-rence at a patch level. The data design focuses on efficiently storing, access-ing, and preprocessing both the environmental features and fire occurrence labels.

  Dataset Structure

  The dataset is organized in a patch-wise and time-aware manner. Each sample corresponds to a square patch of the study area and includes a sequence of temporal snapshots for the input variables.

  – Patch Identification:

  – year: The calendar year of the observation.

  – patch row, patch col: Row and column indices of the patch within the raster grid.

  – Temporal Steps:

  – t1, t2, ..., tN: Time indices for sequential input frames.

  – Each time step corresponds to a 16-day cumulative raster (MODIS-like resolution) for environmental variables.

  – Features:

  – ndvi16: Normalized Difference Vegetation Index, 16-day com-posite.

  – temp16 : Surface temperature, 16-day composite.

  – precip16: Precipitation, 16-day cumulative.

  – rh16 : Relative humidity.

8

– vpd16 : Vapor pressure deficit.

– elevation : Static topography data.

– slope : Derived from elevation.

– Target Variable:

– has fire : Binary label (0 or 1\) indicating whether fire occurred in the patch during the target time step.

Data Representation

Each dataset sample is represented as a tensor of shape:

X ∈ RT×C×H×W ,	y ∈ {0, 1}

where:

– T \= number of input time steps (e.g., 3\)

– C \= number of feature channels (7 original features \+ 1 optional miss-ingness mask)

– H, W \= spatial dimensions of the patch (e.g., 32 × 32 pixels)

– y \= patch-level fire occurrence label

Data Storage and Access

– Raster data is stored in GeoTIFF format, organized by variable and year.

– Static variables (elevation, slope) are stored in a separate directory.

– An index CSV file maps each patch and time step to the corresponding raster files and provides the label.

– The dataset loader efficiently reads patches with:

* masked=True and boundless=True to handle nodata and edge cases

* Fill value for missing pixels

* Optional missingness mask channel for the model to distinguish real zeros from filled zeros

  Summary

  The dataset design supports:

  – Multi-temporal sequence modeling with ConvLSTM

  – Handling of missing data and nodata values

  – Patch-level prediction of fire occurrence

  – Efficient caching and raster reading for large spatial coverage

9

3.4	Implementation Details

3.4.1	Software Implementation

Applicable to software-based and machine learning projects.

3.4.2	Machine Learning Implementation

This subsection is mandatory for machine learning projects.

Dataset Description

* Data Source:

  – Satellite-derived environmental raster data collected over Nepal.

  – Multi-source geospatial products representing environmental con-ditions influencing wildfire occurrence.

  – Fire labels derived from satellite-based active fire detection prod-ucts, indicating fire activity within a spatial region during a future prediction window.

  – The study area is divided into fixed-size spatial patches, with data organized temporally to capture environmental changes over time.

* Dataset Size and Features:

  – Each sample represents a spatio-temporal patch sequence.

  – Input consists of multiple consecutive time steps (e.g., 16-day cu-mulative intervals).

  – Each time step contains multiple environmental feature layers (e.g., vegetation indices and climate-related variables).

  – Feature layers are stacked channel-wise to form a multi-channel spatial input.

  – A binary label is assigned to each patch indicating whether any fire occurred in that patch during the prediction period.

  – Final per-sample representation:

  – Temporal dimension: sequence of past observations.

  – Spatial dimension: fixed-size grid (patch).

  – Channel dimension: stacked environmental variables plus a missing-data mask.

  – This formulation enables the model to learn spatial patterns within each patch and temporal dynamics across time.

10

* Data Preprocessing Techniques:

  – Missing or invalid values are replaced with \-9999.

  – A binary missingness mask is added as an extra channel to explicitly indicate unreliable pixels and the missing values don’t play any role in normalization.

  – All data is spatially aligned and cropped into fixed-size patches and temporal sequences are constructed using sliding windows.

  – Class imbalance between fire and non-fire patches is handled using weighted sampling during training.

  Model Design

  Training and Evaluation

  > Table 3.1: Hyperparameter Comparison of Wildfire Prediction Models

| Model |  |  | Epoch | LR | Val Loss | Val F1 | Val Precision | Val Recall | Notes |
| :---- | :---- | :---- | ----- | ----- | ----- | :---- | ----- | ----- | :---- |
| M1 |  |  | 2 | 0.002 | 0.0930 | 0.6881 | 0.5620 | 0.8869 | best |
| M1 |  | E | 1 | 0.002 | 0.0931 | 0.5898 | 0.4182 | 1.0000 | lower F1 |
|  |  |  |  |  |  |  |  |  |  |
| M2 |  |  | 7 | 0.001 | 0.9605 | 0.5895 | 0.4748 | 0.7773 | Mid performance |
| M2 |  | E | 13 | 0.0005 | 1.1503 | 0.5881 | 0.4987 | 0.7164 | Early stopping |
|  |  |  |  |  |  |  |  |  |  |

  Note:

  M1 \= model with hidden channels \= 64

  M2 \= model with early stopping and hidden channels \= \[128, 64, 32\]

  3.5	Algorithms and Flowcharts

  Wildfire prediction is modeled as a binary time-series classification problem using spatio-temporal satellite data. The prediction algorithm consisted of the following steps:

1. Extract fire events from the MODIS FIRMS active fire dataset and apply spatio-temporal augmentation to generate negative (non-fire) samples.

2. Extract static geographical features (elevation, slope, distance to roads, land cover, vegetation type) using Google Earth Engine.

3. Collect dynamic environmental variables (e.g., NDVI, temperature, rain-fall, soil moisture) over a 16-day temporal window and normalize the resulting temporal sequences.

11

4. Train a ConvLSTM model using Binary Cross-Entropy loss to capture spatio-temporal dependencies.

5. Evaluate model performance using AUC, Precision–Recall, and F1 met-rics.

6. During inference, generate a fire probability score for each patch and classify fire risk as fire or non-fire.

12

Figure 3.1: Flowchart of the ConvLSTM-based wildfire prediction pipeline

13

3.6	Tools and Technologies Used

This section summarizes the major tools, technologies, platforms, and envi-ronments used during the project development.

Machine Learning and Data Processing

* PyTorch and TensorFlow: Used for implementing, training, and eval-uating deep learning models.

* ConvLSTM: Used for wildfire prediction through satellite imagery.

* Google Earth Engine: Used for large-scale geospatial data extraction and preprocessing.

* NASA FIRMS fire: Used as the fire event source. Fire points were filtered by date range and spatial extent, then used to derive supervised labels aligned to the predictor grid and timestamps.

  Datasets

* NDVI16: 16-day NDVI, MODIS MOD13A2, 1 km

* Precip6: 16-day precipitation, CHIRPS/GPM, 1 km

* Slope: from Digital Elevation Model, 1 km

* RHP16: 16-day relative humidity proxy, ERA5-Land, 1 km

* DEM: Digital Elevation Model, SRTM, 1 km

* Temp16: 16-day mean temperature, MODIS MOD11A2 / ERA5-Land, 1 km

* VPD16: 16-day Vapor Pressure Deficit, derived from Temp16 & RHP16, 1 km

* Fire Label: NASA FIRMS MODIS fire dataset

14

3.6.1	Summary of Required Content by Project Type

Table 3.2: Summary of Design and Implementation Content for Current Project

| Section / Component | Software | Machine | Hardware |
| :---- | :---- | :---- | :---- |
|  | Project | Learning | Project |
|  |  | Project (This |  |
|  |  | Work) |  |
| System Overview | Frontend | Wildfire | N/A |
|  | dashboard | prediction |  |
|  | with | pipeline |  |
|  | visualization |  |  |
|  | and APIs |  |  |
|  |  |  |  |
| Software Requirements | React | Python, | N/A |
|  |  | PyTorch, |  |
|  |  | NumPy, |  |
|  |  | Pandas, |  |
|  |  | scikit-learn |  |
|  |  |  |  |
| Hardware Requirements | N/A | GPU for | N/A |
|  |  | training and |  |
|  |  | inference |  |
|  |  |  |  |
| System Architecture | Component- | Training and | N/A |
|  | based | inference |  |
|  | frontend | pipeline |  |
|  |  |  |  |
| Module / Component Design | Frontend | Preprocessing | N/A |
|  | visualization | and evaluation |  |
|  | modules | modules |  |
|  |  |  |  |
| Database / Data Design | JSON for | Dataset | N/A |
|  | fetching | Structure and |  |
|  |  | features |  |
|  |  |  |  |
| Implementation Details | API and UI | Model | N/A |
|  | integration | training |  |
|  |  | inference |  |
|  |  |  |  |
| Algorithms and Flowcharts | N/A | Prediction | N/A |
|  |  | model |  |
|  |  | flowchart |  |
|  |  |  |  |
| Evaluation / Testing | Functional | Validation | N/A |
|  | testing of | metrics: |  |
|  | dashboard | PR-AUC, F1, |  |
|  |  | Precision, |  |
|  |  | Recall and |  |
|  |  | test set |  |
|  |  | evaluation |  |
|  |  |  |  |
| labeltab:current-project-summary |  |  |  |

15

Chapter 4

Results and Discussion

This chapter presents the results obtained from the implementation of the wildfire prediction. It focuses on model performance, key features, challenges encountered, deviations from original objectives, and overall effective- ness.

4.1	Implemented Features

The following key features were successfully implemented:

* Flood Prediction Module : Location-based flood risk prediction using BiLSTM trained on spatio-temporal datasets combining static (terrain, slope, dis- tance to nearest river, TWI, land cover) and dynamic (rainfall) features.

* Web Application Interface: Interface for selecting locations, viewing wildfire mask, and obtaining risk predictions.

* Data Pipeline: Automated preprocessing and feature extraction using Google Earth Engine (GEE), NASA FIRMS resource management sys-tem.

* Visualization and Alerts: Display of Fire Mask

  4.2	Results and Performance Analysis

  4.2.1	Wildfire Prediction System

  :

  Metric	Value

  loss	0.15

  AUC	0.505

  Recall	0.89

16

4.2.2	Evaluation Metrics

:

Confusion Matrix

:

Figure 4.1: Flowchart of the ConvLSTM-based wildfire prediction pipeline

The model correctly classified 1,434 non-fire instances (True Negatives) and 926 fire instances (True Positives). However, it generated 1,231 false alarms (False Positives) and missed 105 actual fires (False Negatives). While the model detects most fires ( 90% recall), the high false positive rate indicates it’s overly sensitive, favoring safety over accuracy by erring on the side of caution.

4.2.3	Precision Recall Curve

:

This value is notably low (close to random performance) and indicates poor overall model performance. A strong classifier would have a PR AUC closer to 1.0. The low score suggests the model struggles to maintain high precision while achieving high recall, confirming the imbalanced performance observed in the confusion matrix.

17

4.2.4	Precision Recall Curve

:

This value is notably low (close to random performance) and indicates poor overall model performance. A strong classifier would have a PR AUC closer to 1.0. The low score suggests the model struggles to maintain high precision

18

while achieving high recall, confirming the imbalanced performance observed in the confusion matrix.

4.3	Challenges and Limitations

This section discusses the major challenges faced during the project execution and the limitations of the implemented system.

Guidelines by project type:

– Software projects: scalability issues, integration problems, performance constraints

– Machine learning projects: data limitations, overfitting, computational constraints

– Hardware projects: component availability, power constraints, envi-ronmental issues

Students should also explain how these challenges were mitigated or why certain limitations remain.

4.4	Discussion

This section provides an overall interpretation of the results. It should connect the outcomes with the project objectives and explain the significance of the findings. Any unexpected results or negative findings should be discussed honestly and critically.

19

Chapter 5

Conclusion and Future Works

This chapter concludes the project by summarizing the overall work carried out and the extent to which the stated objectives were achieved. The achieved and unachieved goals should be discussed with valid technical or practical reasoning. The conclusion must be consistent with the project objectives de-fined in earlier chapters and should not introduce any new results, methods, or claims.

The summary should highlight the major contributions of the project, key outcomes, and lessons learned during implementation. The conclusion should be concise and reflective rather than descriptive.

5.1	Limitations

This section discusses the limitations of the proposed system or approach. Limitations may arise due to constraints related to time, resources, data avail-ability, tools, or scope of the project. The limitations should be stated honestly and clearly.

Guidelines by project type:

– Software projects: scalability issues, limited platform support, perfor-mance constraints

– Machine learning projects: dataset size or quality, model generaliza-tion, computational limitations

– Hardware projects: hardware precision, environmental dependency, component constraints

Example:

– The system was tested only on a limited dataset due to time constraints.

– Real-time performance optimization was not fully explored.

– The solution was validated in a controlled environment only.

20

5.2	Future Enhancements

This section outlines possible improvements and extensions that can be un-dertaken in the future to overcome the identified limitations or to expand the scope of the project. The proposed enhancements should be realistic, techni-cally sound, and aligned with the project domain.

Guidelines by project type:

– Software projects: feature expansion, scalability improvements, de-ployment in real-world settings

– Machine learning projects: larger datasets, advanced models, real-time or multimodal integration

– Hardware projects: improved sensors, enhanced robustness, field de-ployment and testing

Future enhancements should be presented as logical next steps rather than promises or guarantees.

21

References

Li, X., Wang, Y., and Zhang, L. (2021). Geographically weighted logistic regression for forest fire prediction in yunnan province, china. Remote Sensing of Environment, 258:112368.

Prapas, P., Kotsiantis, S., and Karatzas, K. (2023). Deep learning frame-work for daily wildfire danger forecasting in greece. Fire Safety Journal, 134:104634.

Xu, Y., Wang, S., and Li, M. (2025). Advances in deep learning for wild-fire prediction: Methods, challenges, and future directions. International Journal of Wildland Fire, 34:1–22.

Zhang, H., Liu, J., and Chen, Q. (2019). Cnn-based forest fire susceptibility mapping using remote sensing data. In Proceedings of the 2019 IEEE In-ternational Geoscience and Remote Sensing Symposium (IGARSS), pages 1234–1237.

22

Appendix

The appendix contains supplementary material that supports the main content of the report but is not essential to include in the main chapters. Materials placed in the appendix should be referenced at appropriate places in the re-port.

Important Guideline: Only important and representative results, figures, and tables should appear in the main chapters. Detailed, repetitive, or ex-tended materials must be placed in the appendix.

A.1	Additional Figures and Screenshots

This section may include figures that demonstrate intermediate results, system interfaces, experimental outputs, or hardware setups that are too detailed for the main chapters.

Example usage:

– Software project: additional UI screenshots, admin panels, error han-dling views

– Machine learning project: additional plots (loss curves, confusion ma-trices, ROC curves)

– Hardware project: extra circuit diagrams, wiring layouts, hardware setup photographs

All figures must include proper captions and labels.

A.2	Extended Tables and Results

This section includes large or detailed tables that supplement the main results.

Example usage:

– Software project: full feature comparison tables, test case results

– Machine learning project: detailed evaluation metrics, per-class perfor-mance tables

– Hardware project: component specifications, calibration results

23

A.3	Algorithms and Pseudocode

Include detailed algorithms or pseudocode that are referenced but not fully explained in the main chapters.

Example usage:

– Software project: core business logic algorithms

– Machine learning project: training or inference algorithms

– Hardware project: control logic or state transition algorithms

A.4	Source Code Snippets (Optional)

This section may include important code snippets that illustrate implementa-tion logic.

Guidelines:

– Do not include full source code listings in the appendix

– Only include short, well-commented snippets

– Refer to external repositories (e.g., GitHub) for complete code

A.5	User Manuals or Installation Guides (Optional)

Include user instructions, setup procedures, or deployment steps that are use-ful for reproducibility.

Example usage:

– Software project: installation steps, environment setup

– ML project: dataset preparation, model execution steps

– Hardware project: assembly instructions, safety precautions

A.6	Ethical Approval or Consent Forms (If Applicable)

If the project involves human participants, user data, or sensitive information, relevant approval or consent documents should be included here.

A.7	Additional Documentation

Any other supplementary material that does not fit into the above categories may be included here, provided it is clearly labeled and referenced.

24

Appendix Referencing Rule

All appendix items must be referenced in the main text using the format:

“Refer to Appendix-A, Section X for details.”

Unreferenced appendix content may be ignored during evaluation.

25