# ML Models for Optimizing Redox States in NMN-producing MCFs


## Project Overview
This project includes three machine learning models used for predicting the NMN prodution based on the expression intensity of biological components of redox balancing system. It encompasses tasks such as data preprocessing, model training, evaluation, and combinatorial_space_screening, among others. More details can be found below.


## Usage
1. The packages used to train and evaluate the ML models:
   - `ML_NMN_Redox_RandomForest.py`
     - Train and evaluate a RandomForest model for the collected biological data (165 samples)
   - `ML_NMN_Redox_SVR.py`
     - Train and evaluate a Support Vector Regression model for the collected biological data (165 samples)
   - `ML_NMN_Redox_XgBoost.py`
     - Train and evaluate a XgBoost model for the collected biological data (165 samples)
   - `Combinatorial_space_screening.py`
     - Screen the Combinatorial_space by the three ML models to identify the top 1% predictions
     - Visualize the distribution of collected data samples and predicted top 1% combinations in combinatorial space
     - Visualize the coverage of different ML models on top 1% predictions
   - `Combinatorial_space.py`
     - Generate all possible samples in the entire combinatorial space

2. The files that contains the collected biological data and predicted output results:
   - `FileS2.csv`
     - Contain the collected biological data used for model training (165 samples)
     - This file can be found in Supplementary data of publication below:
       - "Metabolic reprogramming and machine learning-guided redox balancing to boost production of nicotinamide mononucleotide"
   - `Combinatorial_space.csv`
     - Contain all possible samples in the entire combinatorial space (1700 samples)
## License
This project is licensed under the MIT License.


## Contact
For any questions or issues, please open contact:
- Hao Song: hsong@tju.edu.cn
- Bo Xiong: rookie_b0@tju.edu.cn
