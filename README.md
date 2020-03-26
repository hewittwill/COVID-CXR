# COVID-CXR

The  COVID-19 pandemic is having a devastating effect on public health worldwide. With no vaccine or treatment available, the key method to prevent the spread of the infection is mass-screening, containment and contact-tracing. This method of infection control relies on our ability to rapidly, efficiently and economically screen patients for disease. 

Chest X-Ray (CXR) has been proposed as a rapid triaging tool for patients presenting with clinical symptoms associated with COVID-19 (fever, dry cough and shortness of breath), and to rule out COVID-19 to allow the patient to be put into a lower degree of isolation. 

This repository is a collaborative effort to make a publicly available dataset and propose a convolutional neural network (CNN) to rapidly distinguish between COVID-19, Other Viral Pneumonia, Bacterial Pneumonia and Normal chest x-rays. 

## Dataset

This repository uses git-lfs to host and track changes in the dataset. The dataset is prepared using the `prepare_data.py` script, although should not be necessary as cloning the repository will pull the dataset as well. The dataset is a combination of two publicly available datasets, [ieee8023's covid-chest-xray-dataset repository](https://github.com/ieee8023/covid-chestxray-dataset) and the [Kaggle pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of paedatric patients 