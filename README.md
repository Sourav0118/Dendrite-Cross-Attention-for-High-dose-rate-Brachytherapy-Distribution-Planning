# Dose Prediction Using DiCA-UNet for internal beam radiation tharepy

This repository contains the code for predicting radiation dose maps for High-Dose Rate (HDR) Brachytherapy using a DiCA-UNet architecture with a feature extraction approach. This problem is part of an effort to optimize and predict the distribution of radiation dose for cervical cancer treatments, allowing clinicians to personalize therapy based on patient anatomy while ensuring safety for surrounding organs. This repo has the official implementation of the DiCA-UNet architecture proposed in - 

>#### Title: [Dendrite Cross Attention for High Dose Rate Brachytherapy Distribution Planning](https://github.com/Sourav0118/Dendrite-Cross-Attention-for-High-dose-rate-Brachytherapy-Distribution-Planning) (To Be Updated!)
> ##### Authors: [Souarv Saini](https://scholar.google.com/citations?hl=en&user=r_NVq3IAAAAJ&view_op=list_works&gmla=AOAOcb2PwAT-WhcOFo33z3wArFzRwQYnAWvt8sY8tBA9ASJ4pVJOY9nFRY7D0TPjY698ITJHugLs3-oDgY6wiTAHgGjX_JKJ9jIZ9656K-Sx5lhngQS2gmYTUNs21Whmqloc0CaJQJc), [Zhen Li](https://orcid.org/0000-0002-3769-8612), [Xiaofeng Liu](https://scholar.google.com/citations?user=VighnTUAAAAJ&hl=en)

---

## Task Description

In High-Dose-Rate (HDR) Brachytherapy, radiation is delivered directly to the tumor site using various types of applicators. These applicators can vary in shape and design, making dose distribution prediction a complex task. Precise prediction of the radiation dose distribution is critical to ensure that the tumor receives the required dose while minimizing exposure to adjacent organs such as the bladder and rectum.

The challenge in HDR Brachytherapy lies in generalizing the dose map prediction across different applicator types. Traditional methods may struggle to accurately predict dose distributions for new or varied applicator designs. This work aims to address this issue by employing deep learning models to automate dose map prediction. By training on diverse datasets and generalizing across different applicator types, the proposed approach seeks to improve the accuracy and efficiency of treatment planning.

### References

- [HDR Brachytherapy: Overview and Techniques](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6723180/)
- [Applicator Types and Dose Distribution in Brachytherapy](https://www.sciencedirect.com/science/article/pii/S0360301619301667)
- [Machine Learning for Medical Image Analysis in Brachytherapy](https://www.journalofmedicalimaging.org/doi/10.1117/1.JMI.6.2.021001)

---

### Objective

The goal is to develop a deep learning model that can predict the dose distribution from CT scans and organ segmentations (CTV, bladder, rectum) with high accuracy.

---

## Dataset

The dataset used for this project contains CT scans of patients undergoing brachytherapy treatment. Each patient case includes:

![alt text](https://github.com/Sourav0118/Dendrite-Cross-Attention-for-High-dose-rate-Brachytherapy-Distribution-Planning/blob/main/eccvw.png?raw=True)

- **CT Scan (CT):** Imaging data of the patient's anatomy.
- **CTV Segmentation:** Clinical Target Volume, indicating the tumor area.
- **Bladder Segmentation:** Bladder area segmentation.
- **Rectum Segmentation:** Rectum area segmentation.
- **Applicator Segmentation:** The applicator's position within the body for delivering the radiation.
- **Dose Map:** Ground truth radiation dose delivered to the tumor and surrounding areas.

- Here's the revised section for the README with a single data directory:

---

### Dataset Structure

The dataset is organized into a single directory containing multiple subdirectories for individual cases. Each case includes several NIfTI files corresponding to different imaging modalities and segmentations. The directory structure is as follows:

```
/data 
 │
 ├───1785 
 │      └───ct.nii.gz
 │      └───ctv.nii.gz
 │      └───bladder.nii.gz
 │      └───rectum.nii.gz
 │      └───dose.nii.gz
 │
 ├───1759
 │      └───ct.nii.gz
 │      └───ctv.nii.gz
 │      └───bladder.nii.gz
 │      └───rectum.nii.gz
 │      └───dose.nii.gz
 │
 ├───...
```

Please ensure that the Patient ID, which is a subfolder of the data, is unique for each patient. The dataset is divided into training and validation sets, with an 80:20 split.

---

## Model Architecture

![alt text](https://github.com/Sourav0118/Dendrite-Cross-Attention-for-High-dose-rate-Brachytherapy-Distribution-Planning/blob/main/Multi-Unet-with-scans.png?raw=True)

The model used consists of two main components:

1. **Feature Extractor:** A Dendrite-like U-Net architecture that processes the CT scans and organ segmentations to extract features.
2. **Label Predictor:** A smaller network that takes the extracted features and predicts the radiation dose map.

---

## Training and Inference

### Requirements

To run the code, ensure you have the following libraries installed:

- PyTorch
- TorchVision
- NumPy
- nibabel (for handling `.nii.gz` files)
- tqdm
- WandB (Weights and Biases for experiment tracking)

You can install the necessary packages via:

```bash
pip install -r requirements.txt
```

### Training

To train the model on your dataset, use the following command:

```bash
python main.py --data_path <path_to_dataset> --ckpt_path <path_to_checkpoint> --batch_size <batch_size> --num_epochs <num_epochs> --lr <learning_rate> --device <cuda/cpu> --use_wandb <1/0>
```

- **data_path:** Path to the dataset (which should include CT, CTV, bladder, rectum, and dose maps in `.nii.gz` format).
- **ckpt_path:** Path to store the model checkpoints.
- **batch_size:** Batch size for training.
- **num_epochs:** Number of training epochs.
- **lr:** Learning rate for the optimizer.
- **device:** Set to `cuda` if using a GPU, otherwise `cpu`.
- **use_wandb:** Set to `1` to enable logging with WandB, or `0` to disable it.

### Inference

For running inference on new test data, use the `inference.py` script:

```bash
python inference.py --model_path <path_to_model> --input_dir <path_to_test_data> --output_dir <path_to_save_results> --use_wandb <1/0>
```

- **model_path:** Path to the trained model weights.
- **input_dir:** Directory containing the test data in `.nii.gz` format.
- **output_dir:** Directory to save the predicted dose maps.
- **use_wandb:** Enable WandB logging for inference by setting this to `1`.

### Example Usage

```bash
# For training
python main.py --data_path ./dataset --ckpt_path ./checkpoints --batch_size 3 --num_epochs 200 --lr 1e-5 --device cuda --use_wandb 1

# For inference
python inference.py --model_path ./checkpoints/model.pth --input_dir ./test_data --output_dir ./output --use_wandb 1
```

### Checkpointing and Resuming Training

The model automatically saves checkpoints during training. If the training is interrupted or you wish to resume from a previous checkpoint, ensure the model weights are stored at the `ckpt_path`. The script will load the latest saved weights and continue training.

---

## Logging and Visualization with WandB

Weights and Biases (WandB) is used for experiment tracking, model logging, and performance visualization. If enabled, each training run logs metrics like training and validation loss, and stores model checkpoints.

To enable WandB, use the `--use_wandb 1` flag during training and inference.

If you want to track experiments for team-based projects, you can specify a WandB entity with the `--wandb_entity <entity_name>` argument.

---

## Model Evaluation

During training, the model is evaluated based on the weighted average of Masked Root Mean Squared Error (masked RMSE) between the predicted dose map and the ground truth dose map over masked regions of CTV, bladder, rectum, and applicator. The best-performing model is saved based on validation loss.

## Citations

```
to be updated.
```

---

## Contribution and Issues

Feel free to raise issues or contribute to improving the code. You can fork this repository, make your changes, and submit a pull request. We welcome contributions in the form of bug fixes, code improvements, or even new features like additional metrics for model evaluation.

