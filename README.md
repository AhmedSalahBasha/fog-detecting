# Detection of Movement Disorder using Machine Learning

## Introduction:
This is the codebase repository of the Master Thesis "Detection of Movement Disorders In Parkinson's Disease Using 
Machine Learning Techniques" at The Technical University of Berlin in January 2021. The main target is build an 
automated system that can detect Freezing of Gait (FOG) events using inertial data extracted by wearable sensors 
(Accelerometer and Gyroscope) on the left and right shank and feet of the patient. <br/>
The dataset contains data from 16 patients who have done different lab experiments at Charité – Universitätsmedizin Berlin.

## Prerequisites: 
1. Install new python virtual environment: <br/>
`$ virtualenv <env_name>` <br/>
2. Activate the virtual environment: <br/>
`$ source <env_name>/bin/activat`
3. Install the requirements libraries: <br/>
`(<env_name>)$ pip install -r path/to/requirements.txt`

## Project Structure: 
Here is the directory tree of the project: 

    |-- cleaning
    |   `-- cleaning.py
    |-- experiments
    |   |-- loocv.py
    |   |-- train_test.py
    |   `-- window_size_experiments.py
    |-- modelling
    |   |-- dl_models.py
    |   |-- ml_models.py
    |   `-- modelling.py
    |-- preprocessing
    |   |-- features_selection.py
    |   |-- preprocessing.py
    |   `-- rolling_window.py
    |-- tuning
    |   |-- feature_importance.py
    |   `-- lstm_grid_search.py
    |-- README.md
    `-- requirements.txt

- `cleaning` direcory contains the script the does the cleaning process.
- `preprocessing` directory which contains three python files:
    - `preprocessing.py` file that is the main file for the preprocessing pipeline.
    - `rolling_window.py` file that does the rolling window technique for features extraction.
    - `features_selection.py` file that filters the features based on specific conditions according to the experiments.
- `modelling` directory which contains three python scripts:
    - `dl_models.py` that contains the classes for the deep neural networks models.
    - `ml_models.py` that contains the classes for the classic machine learning models.
    - `modelling.py` that is the main file for the modelling pipeline.
- `tuning` directory contains two python scripts for models tuning:
    - `feature_importance.py` that extracts the information about the features importance to the model.
    - `lstm_grid_search.py` that does the Grid-Search Cross-Validation technique for the LSTM model.
- `experiments` directory contains three python scripts which is main three experiments:
    - `train_test.py` is the first experiment for finding the best sensor, sensor location, features and model that 
    yields highest model evaluation score.
    - `window_size_experiment.py` is the second experiment for finding the best rolling window size that yields the 
    highest model evaluation score.
    - `loocv.py` is the Leave-one-out Cross-validation experiment for evaluating the best model against every single 
    patient independently.
    
## Acknowledgement:
This work is implemented by Ahmed Basha as a part of his Master Thesis for the department of the Control Systems at The 
Technical University of Berlin under a supervision of Prof.Dr. Thomas Schauer ,Prof.Dr.Ing. Clemens Gühmann 
and the assistant supervisor Ardit Dvorani. 

@TU-Berlin | 2021