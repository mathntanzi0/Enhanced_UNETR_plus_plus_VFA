# VFA UNETR++: Deep Learning-Based Automatic Segmentation of 3D Medical Images

Welcome to the **VFA UNETR++** repository, which provides deep learning models optimized for high-precision 3D medical image segmentation. This repository contains two primary model variants, each with **native (Python-only)** and **CUDA-optimized** implementations for enhanced computational efficiency.

## Model Variants and Implementations
![epa_block](https://github.com/user-attachments/assets/310150f6-11e1-4fef-89ce-f2b848d01900)
Overview of the Enhanced Efficient Paired-Attention Block with the spatial branch at the top and the channel
branch at the bottom. Left: The spatial branch is enhanced to use the Full Voxel-Focused Attention (FVFA) mechanism.
Right: The spatial branch is enhanced to use the Semi Voxel-Focused Attention (VFA) mechanism.

The following implementations are included:

1. **UNETR++ VFA Native** (Python-only)
2. **UNETR++ VFA** (CUDA-optimized)
3. **UNETR++ FVFA Native** (Python-only)
4. **UNETR++ FVFA** (CUDA-optimized)

## Implementation Details

- **Programming Language**: Python 3.8.19
- **Framework**: PyTorch 2.4.0 with MONAI libraries for medical imaging
- **GPU**: Nvidia V100 16GB (PCIe)
- **CUDA Optimization**: A custom CUDA implementation computes query-key similarity and aggregates attention weights for improved efficiency, compiled with GCC 9.2.0 and CUDA 12.4.

## Datasets

The models are trained and validated on the following datasets:

- **Tumor Dataset**
- **Synapse Dataset**
- **ACDC Dataset**

Each dataset has a dedicated directory structure, with tailored components for segmentation tasks.

## Dataset Organization

Dataset preprocessing follows the steps outlined in [nnFormer](https://github.com/282857341/nnFormer) and [UNETR++](https://github.com/Amshaker/unetr_plus_plus). The preprocessed datasets for Synapse, ACDC, and BRaTs are organized as follows:


### BRaTs Dataset
The folder structure for the BRaTs dataset should be as follows:

```
./DATASET_Tumor/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task003_tumor
       ├── unetr_pp_cropped_data/
           ├── Task003_tumor
 ```

### Synapse Dataset
The folder structure for the Synapse dataset should be as follows:

```
./DATASET_Synapse/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task002_Synapse
       ├── unetr_pp_cropped_data/
           ├── Task002_Synapse
 ```
### ACDC Dataset
The folder structure for the ACDC dataset should be as follows:

```
./DATASET_Acdc/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task001_ACDC
       ├── unetr_pp_cropped_data/
           ├── Task001_ACDC
 ```


Please refer to the [UNETR++ repository](https://github.com/Amshaker/unetr_plus_plus) to download the preprocessed datasets and follow their instructions.

## Directory Structure and Key Files

Each model variant (`unetr_plus_plus_VFA_native`, `unetr_plus_plus_VFA`, `unetr_plus_plus_FVFA_native`, and `unetr_plus_plus_FVFA`) includes dataset-specific files organized as follows:

### Tumor Dataset
- **transformerblock**: Contains the custom transformer block for Tumor dataset segmentation.
- **model_components**: Layers and components specialized for the Tumor dataset.
- **unetr_pp_tumor.py**: Main implementation file for the Tumor dataset.

### Synapse Dataset
- **transformerblock**: Transformer block adapted for Synapse dataset segmentation.
- **model_components**: Segmentation components tailored for the Synapse dataset.
- **unetr_pp_synapse.py**: Primary implementation file for the Synapse dataset.

### ACDC Dataset
- **transformerblock**: Transformer block designed for ACDC dataset segmentation.
- **model_components**: Custom layers for the ACDC dataset.
- **unetr_pp_acdc.py**: Main script for ACDC segmentation.


## Evaluation

To reproduce the results of the models:

1. Download the models' weights from the following folder: [VFA UNETR++ weights folder](https://drive.google.com/drive/folders/1wc1g7aB0DO1ZIj5pIMpTdo04_68Aj3g-).
   
   - Place the `model_final_checkpoint.model` in the corresponding folder:
   ```shell
   unetr_pp/evaluation/unetr_pp_acdc_checkpoint/unetr_pp/3d_fullres/Task001_ACDC/unetr_pp_trainer_acdc__unetr_pp_Plansv2.1/fold_0/
   ```
   Then, run:
   ```shell
   bash evaluation_scripts/run_evaluation_acdc.sh
   ```

2. Download the Synapse weights from the folder linked above, and paste `model_final_checkpoint.model` in the following path:
   ```shell
   unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/
   ```
   Then, run:
   ```shell
   python unetr_pp/inference/predict_simple.py
   python unetr_pp/inference_synapse.py
   ```

3. Download the BRaTs weights from the folder linked above, and paste `model_final_checkpoint.model` in the following path:
   ```shell
   unetr_pp/evaluation/unetr_pp_lung_checkpoint/unetr_pp/3d_fullres/Task003_tumor/unetr_pp_trainer_tumor__unetr_pp_Plansv2.1/fold_0/
   ```
   Then, run:
   ```shell
   python unetr_pp/inference/predict_simple.py
   python unetr_pp/inference_tumor.py
   ```

