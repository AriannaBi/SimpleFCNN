## DATA AUGMENTATION FOR SENSOR-BASED SLEEP AND WAKE RECOGNITION

The project in Data Augmentation folder contains the code to read the folders of each user, create the dataset, augment it and train the network models.

- Folders:
-- plots: contains the code to create the plots and the plots obtained;
-- datasets: contain the code to read the sessions and create a datase;

- Files:
- - EDA_model.py contain the code to train and evaluate the EDA network model
- - BVP_model.py contain the code to train and evaluate the BVP network model
- - ACC_model.py contain the code to train and evaluate the ACC network model
- - TEM_model.py contain the code to train and evaluate the TEM network model

- - augment_data.py is the code that takes the dataset and augment it of 50\%.

- - DataAugmentation_TimeseriesData.py is the adapt code of Um. 


To create the dataset, run  " python3 datasets/create_datasets.py ";

To run EDA network model, run " python3 EDA_model.py ";

To run BVP network model, run " python3 BVP_model.py ";

To run ACC network model, run " python3 ACC_model.py ";

To run TEM network model, run " python3 TEM_model.py ";

To create the plots, run " python3 plots/Plots.ipynb "

The dataset read from folder "Sessions" which contains all the user with the relative sessions, so it is important that a folder "Sessions" exists. In this gitHub repo there is not enough space for it. 

The results are included in the folder "Results"
