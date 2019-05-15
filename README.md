# Predict-Lung-Disease-through-Chest-X-Ray
We obtain this repository by refactoring the [code](https://github.com/Azure/AzureChestXRay) for the blog post [Using Microsoft AI to Build a Lung-Disease Prediction Model using Chest X-Ray Images](https://blogs.technet.microsoft.com/machinelearning/2018/03/07/using-microsoft-ai-to-build-a-lung-disease-prediction-model-using-chest-x-ray-images/). This instruction aims to help newcomers build the system in a very short time.   
# Installation
1. Clone this repository
   ```Shell
   git clone https://github.com/svishwa/crowdcount-mcnn.git
   ```
   We'll call the directory that you cloned PredictLungDisease `ROOT`  
  
2. All essential dependencies should be installed  
# Data set up
1. Download the NIH Chest X-ray Dataset from here:  
   https://nihcc.app.box.com/v/ChestXray-NIHCC.  
   You need to get all the image files (all the files under `images` folder in NIH Dataset), `Data_Entry_2017.csv` file, as well as the      Bounding Box data `BBox_List_2017.csv`.  

2. Create Directory 
   ```Shell
   mkdir ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC
   mkdir ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other
   ```  
3. Save all images under `ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC`  

4. Save `Data_Entry_2017.csv` and `BBox_List_2017.csv` under `ROOT/azure-share/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other`  

5. Process the Data
   ```Shell
   mkdir ROOT/azure-share/chestxray/output/data_partitions
   ```  
   Run `000_preprocess.py` to create `*.pickle` file under this directory 
# Test  
1. We have provided the pretrained-model `azure_chest_xray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5` under `ROOT/azure-  share/chestxray/output/fully_trained_models`
