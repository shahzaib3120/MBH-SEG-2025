import subprocess
import shutil
import torch
import os
import numpy as np
from scipy.ndimage import label
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
import argparse
from totalsegmentator.python_api import totalsegmentator

def move_and_rename_nifti_files(parent_folder):
    # Loop through all subdirectories in the parent folder
    for subdir in os.listdir(parent_folder):
        subdir_path = os.path.join(parent_folder, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Loop through files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.nii.gz'):
                    file_path = os.path.join(subdir_path, filename)

                    # Construct new filename based on the parent directory name
                    new_filename = f"{subdir}.nii.gz"
                    new_file_path = os.path.join(parent_folder, new_filename).replace("_0000.nii.gz", ".nii.gz")

                    # Move and rename the file one directory up
                    shutil.move(file_path, new_file_path)
                    print(f"Moved and renamed: {file_path} -> {new_file_path}")

            os.rmdir(subdir_path)


def combine_segmentations(folder1, folder2, output_folder, nifti_dir):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folder1_files = [f for f in os.listdir(folder1) if f.endswith('.nii.gz')]
    folder2_files = [f for f in os.listdir(folder2) if f.endswith('.nii.gz')]

    # Ensure both folders contain the same files
    common_files = set(folder1_files).intersection(set(folder2_files))

    if not common_files:
        print("No common .nii.gz files found between the folders.")
        return

    for file in common_files:
        file1_path = os.path.join(folder1, file)
        file2_path = os.path.join(folder2, file)

        try:
            corresponding_nifti_path = os.path.join(nifti_dir, file.replace(".nii.gz", "_0000.nii.gz"))
            nifti_img = nib.load(corresponding_nifti_path)

            # Load both NIfTI files
            nifti1 = nib.load(file1_path)
            nifti2 = nib.load(file2_path)

            # Extract the voxel data from both files
            data1 = nifti1.get_fdata()
            data2 = nifti2.get_fdata()

            # Combine segmentations: retain non-zero voxels from both
            combined_data = np.maximum(data1, data2)

            # Create a new NIfTI image with the combined data
            combined_nifti = nib.Nifti1Image(combined_data, nifti_img.affine, nifti_img.header)

            # Save the combined NIfTI image to the output folder
            output_path = os.path.join(output_folder, file)
            nib.save(combined_nifti, output_path.replace(".nii.gz", "_0001.nii.gz"))

            print(f"Combined segmentation saved for {file}.")

        except Exception as e:
            print(f"Error processing {file}: {e}")

def binarize_charite_outputs(segmentation_directory, nifti_dir):
    for file_name in os.listdir(segmentation_directory):
        if file_name.endswith(".nii.gz"):

            corresponding_nifti_path = os.path.join(nifti_dir, file_name.replace(".nii.gz", "_0000.nii.gz"))

            seg_file_path = os.path.join(segmentation_directory, file_name)

            # Load the NIfTI file
            nifti_img = nib.load(corresponding_nifti_path)
            seg_img = nib.load(seg_file_path)

            # Get the data from the NIfTI file
            nifti_data = nifti_img.get_fdata()
            seg_data = seg_img.get_fdata()

            print(f"Processing file: {seg_file_path}")
            print(f"Shape of NIfTI array: {nifti_data.shape}")

            # Initialize a new array to hold the mapped labels
            new_nifti_data = np.zeros_like(nifti_data)

            # Mapping old labels to new labels
            label_mapping = {
                1: 1,  # SAH
                2: 1,  # IVHem
                4: 1,  # ICH
                6: 1,  # subdural
                5: 1,  # aneurysm
                3: 0,  # ventricle to 0
            }

            # Apply the label mapping to swap labels in the new array
            for old_label, new_label in label_mapping.items():
                new_nifti_data[seg_data == old_label] = new_label

            # Print the unique values in the updated NIfTI data for verification
            print("Unique labels in the updated NIfTI data:", np.unique(new_nifti_data))

            # Create a new NIfTI image from the updated data
            new_nifti_img = nib.Nifti1Image(new_nifti_data, affine=nifti_img.affine)

            output_nifti_path = seg_file_path
            nib.save(new_nifti_img, output_nifti_path)


def total_segmentator_segment(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_filepath = os.path.join(input_folder, filename)

            # Construct the output path (without the .nii.gz extension)
            output_name = filename.replace('.nii.gz', '')
            output_filepath = os.path.join(output_folder, output_name)

            # Construct the TotalSegmentator command
            command = [
                "TotalSegmentator",
                "-i", input_filepath,
                "-o", output_filepath,
                "-ta", "cerebral_bleed"
            ]

            # Execute the command using subprocess
            try:
                subprocess.run(command, check=True)
                print(f"Successfully processed: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {str(e)}")


def add_channel_back_to_folder(output, original_test_folder):
    for file in os.listdir(output):
        if ".nii" in file:
            file_path = os.path.join(output, file)

            shutil.copy(file_path, original_test_folder)

def run_nnUNet_ensemble(output_folder_2d, output_folder_3d, results_folder):
    # Define the command as a list of arguments
    command = [
        "nnUNetv2_ensemble",
        "-i",
        output_folder_2d,
        output_folder_3d,
        "-o",
        results_folder,
        "-np",
        "8"
    ]

    # Run the command and capture the output
    result = subprocess.run(command, check=True, text=True, capture_output=True)

def presegmentation(total_segmentator_output_folder, charite_results_folder, final_channel_folder, imagesTs_path):
    predictor_2d = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor_3d = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # CHARITE 2d
    predictor_2d.initialize_from_trained_model_folder(
        "models/prelabeling/nnUNetTrainer__nnUNetPlans__2d",
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor_3d.initialize_from_trained_model_folder(
        "models/prelabeling/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor_2d.predict_from_files(imagesTs_path,
                                 output_folder_2d,
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=8, num_processes_segmentation_export=8,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    predictor_3d.predict_from_files(imagesTs_path,
                                 output_folder_3d,
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=8, num_processes_segmentation_export=8,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # Ensemble between 2d and 3d Charite model
    run_nnUNet_ensemble(output_folder_2d, output_folder_3d, charite_results_folder)

    # binarize multiclass Charite model outputs
    binarize_charite_outputs(charite_results_folder, imagesTs_path)

    # Call the function
    total_segmentator_segment(imagesTs_path, total_segmentator_output_folder)

    # change filenames to nnUnet
    move_and_rename_nifti_files(total_segmentator_output_folder)

    combine_segmentations(total_segmentator_output_folder, charite_results_folder, final_channel_folder, imagesTs_path)



def run_inference(input_folder, output_folder_L):

    predictor_L = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor_L.initialize_from_trained_model_folder(
        "models/multiclass/nnUNetTrainerDA5__nnUNetResEncUNetLPlans__3d_fullres",
        use_folds=(0,1,2,3,4),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor_L.predict_from_files(input_folder,
                                 output_folder_L,
                                 save_probabilities=True, overwrite=False,
                                 folder_with_segs_from_prev_stage=None)


def preprocess_SHD_EDH_in_nifti(nifti_path):
    # Load the NIfTI file
    nifti_file = nib.load(nifti_path)

    print(nifti_path)

    segmentation = nifti_file.get_fdata().astype(np.int32)

    # Define classes
    class_1 = 1
    class_4 = 4
    class_5 = 5

    # Check if the image contains class 1 or class 5
    contains_class_1 = np.any(segmentation == class_1)
    contains_class_5 = np.any(segmentation == class_5)

    # Proceed only if both classes are present
    if contains_class_1 and contains_class_5:
        # Create a mask for class 1 and class 5
        mask = (segmentation == class_1) | (segmentation == class_5)

        # Label connected components in the mask
        labeled_mask, num_features = label(mask)

        # Iterate through each connected component
        for i in range(1, num_features + 1):
            component = (labeled_mask == i)
            unique_classes, counts = np.unique(segmentation[component], return_counts=True)

            # Only interested in components containing class 1 and class 5
            if class_1 in unique_classes and class_5 in unique_classes:
                # Determine the majority class within the component
                majority_class = unique_classes[np.argmax(counts)]

                # Assign the majority class to the whole component
                segmentation[component] = majority_class

    # After processing, count voxels in classes 1 and 5
    class_1_voxels = np.sum(segmentation == class_1)
    class_4_voxels = np.sum(segmentation == class_4)
    class_5_voxels = np.sum(segmentation == class_5)


    # Remove class voxels if they are fewer than 25
    if class_1_voxels < 25 and class_1_voxels !=0:
        segmentation[segmentation == class_1] = 0
        print(f"Class 1 had fewer than 25 voxels, removed all class 1 voxels.")

    if class_4_voxels < 25 and class_4_voxels != 0:
        segmentation[segmentation == class_4] = 0
        print(f"Class 4 had fewer than 25 voxels, removed all class 4 voxels.")

    if class_5_voxels < 25 and class_5_voxels !=0:
        segmentation[segmentation == class_5] = 0
        print(f"Class 5 had fewer than 25 voxels, removed all class 5 voxels.")


    # Save the processed segmentation to a new NIfTI file
    processed_nifti = nib.Nifti1Image(segmentation, nifti_file.affine, nifti_file.header)
    nib.save(processed_nifti, nifti_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nnUNet inference and preprocessing on NIFTI files.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder with NIFTI files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to store the output predictions.')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder


    temp_input_folder = os.path.join(output_folder, "temp", "input")
    shutil.copytree(input_folder, temp_input_folder)

    # Rename niftis to include 0000 for nnUnet format
    for file in os.listdir(temp_input_folder):
        if ".nii" in file:
            file_path = os.path.join(temp_input_folder, file)
            os.rename(file_path, file_path.replace(".nii", "_0000.nii"))


    # Output folders for 3 open source prelabeling models
    output_folder_2d =  os.path.join(output_folder, "temp", "_charite_model_2d")
    output_folder_3d = os.path.join(output_folder, "temp", "_charite_model_3d_fullres")
    total_segmentator_output_folder = os.path.join(output_folder, "temp", "_demo_totalsegmentator")
    charite_results_folder = os.path.join(output_folder, "temp", "ensemble_charite_prelabeling")

    # Final binary prelabeling prediction will be stored in final_channel_folder
    final_channel_folder = os.path.join(output_folder, "temp", "FINAL_PRELABELING_CHANNEL")

    presegmentation(total_segmentator_output_folder, charite_results_folder, final_channel_folder, temp_input_folder)

    add_channel_back_to_folder(final_channel_folder, temp_input_folder)

    torch.multiprocessing.set_start_method('spawn', force=True)

    output_folder_L = os.path.join(output_folder, "temp", "model_L")

    final_results_folder = os.path.join(output_folder, "SecondStage_results/Output")

    run_inference(temp_input_folder, final_results_folder)

    shutil.rmtree(os.path.join(output_folder, "temp"))

    for file in os.listdir(final_results_folder):
        if file.endswith(".nii.gz"):
            nifti_path_to_process = os.path.join(final_results_folder, file)
            preprocess_SHD_EDH_in_nifti(nifti_path_to_process)
        else:
            os.remove(os.path.join(final_results_folder, file))