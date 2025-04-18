# LUNA25 Nodule Block Extraction
This documentation outlines the steps and scripts required to extract and preprocess nodule blocks from CT images for use in training machine learning models. The preprocessing pipeline consists of two main scripts:
1. `extract_nodule_blocks.py` - Extracts 3D nodule patches from CT volumes.
2. `convert2nodule_block.py` - Converts the extracted patches to NumPy format for fast loading during training.

## 1. Nodule Block Extraction
This script extracts 3D image volumes centered around annotated nodules
### Parameters:
- csv_path: Path to the annotations CSV.
- image_root: Path to the folder with original .mha CT scans.
- output_path: Directory where the extracted patches will be saved.
- patch_size: Size of the 3D patch (default: [128, 128, 64]).
- save_format: Output format (default: .nii.gz).

## 2. Convert to Numpy Blocks
This script converts .nii.gz files into .npy format for fast loading.

Required Arguments:
- `--csv_path`: Path to the annotation CSV file.
- `--save_path`: Directory to save the NumPy files.
- `--data_path`: Path to the .nii.gz nodule blocks.
- `--num_workers`: (Optional) Number of parallel workers (default: 8) 