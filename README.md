# Mbh-Seg Solution - Charité Lab for AI in Medicine

## Description
Inference repository for the winning solution in the MICCAI 2024 - MBHSEG Mulitclass Brain Hemorrhage Segmentation Challenge

Challenge Website: https://mbh-seg.com  
[Leaderboard](https://docs.google.com/spreadsheets/d/1yd86k9cPpW6DgKhS3bKsvQ1y5hBrFdEKDAxnG2zrWG8/edit?gid=1447985631#gid=1447985631)

Developed by the Charité Lab for AI in Medicine (CLAIM) research group at Charité University Hospital, Berlin, main developer and person to contact: Orhun Utku Aydin (orhun-utku.aydin@charite.de)

### Please cite the 4 articles in the references section if you use this model in your research 

### Model Details

1. **Input:**
   - TotalSegmentator segmentations serve as an additional input channel to nnUNet models.
   - Open source models from CLAIM contribute to the additional input channel to nnUNet models. (https://github.com/claim-berlin/aSAH-multiclass-segmentation)

2. **Models Used:**
   - **ResEncL** variation of nnUNet models is utilized: 
   -  trained using a 5-fold cross-validation approach.
   - **DA5 Trainer** is employed to enable more extensive data augmentation, while default nnUNet losses and hyperparameters are retained.

### Postprocessing Steps

- **Connected Component Processing:**
  - All voxels of connected components that contain both Subdural and Epidural hemorrhage are reassigned to the predominant hemorrhage class.

- **Small Hemorrhage Removal:**
  - If a segmentation contains fewer than 25 voxels of Subarachnoid (SAH), Epidural, or Subdural hemorrhage, they are excluded and reassigned to the background class.

## Inference

To set up the environment and install the necessary dependencies, follow these steps:

#### Build conda environment
1. Create and Activate a Virtual Environment  
```bash
conda create -n MBHSEG python==3.11.10   
conda activate MBHSEG  
 ```

2. Install the requirements
```bash
cd MBH-SEG-2024-winning-solution
pip install -r requirements.txt
```
#### Download model weights
Download and place models inside **models** folder:
- models/multiclass
- models/prelabeling

Download model weights from:
https://drive.google.com/file/d/1BlE__48PUFbn161nk2XVbHGX0rEK74eX/view?usp=drive_link

### Running inference
Make sure that the conda environment is active
```bash
conda conda activate MBHSEG  
 ```

Run the submission.py specifying an input folder and output folder  
Generic: 

```bash
python submission.py --input_folder absolute_path_to_some_folder_containing_niftis --output_folder absolute_path_to_desired_output_folder
```  

Example on CLAIMs local computer:
```bash
python submission.py --input_folder /media/CLAIM/storage_4tb/Submission_BHDS/folder_to_predict --output_folder /media/CLAIM/storage_4tb/folder_to_predict_segmented
```  

### References
- Julia Kiewitz, Orhun Utku Aydin, Adam Hilbert, Marie Gultom, Anouar Nouri, Ahmed A Khalil, Peter Vajkoczy, Satoru Tanioka, Fujimaro Ishida, Nora F. Dengler, Dietmar Frey
medRxiv 2024.06.24.24309431; doi: https://doi.org/10.1101/2024.06.24.24309431 
- Wu, Biao & Xie, Yutong & Zhang, Zeyu & Ge, Jinchao & Yaxley, Kaspar & Bahadir, Suzan & Wu, Qi & Liu, Yifan & To, Minh-Son. (2023). BHSD: A 3D Multi-Class Brain Hemorrhage Segmentation Dataset. 10.48550/arXiv.2308.11298. 
- Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images
Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, and Martin Segeroth
Radiology: Artificial Intelligence 2023 5:5


