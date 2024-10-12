# ControlNet Skin Tone Dataset Training

This repository contains an example notebook to train a ControlNet model with a custom dataset of skin tones. The notebook includes code to create masks, extract skin tones, generate prompts, and train the model. The notebook is located at `examples/controlnet/main.ipynb`.

## Prerequisites

Before running the notebook, ensure that you have the necessary environment set up.

### 1. Update File Paths
Ensure that all file paths used in the notebook (e.g., for images and masks) are correctly configured to point to the appropriate directories on your system.

### 2. Create the 'crops' Folder
Create a folder named `crops` with approximately 12,000 image files. The images should follow the naming convention:
```
0001.png, 0001_mask.png, ..., 11994.png, 11994_mask.png
```

### 3. Diffusers Repository
Obtain a modified version of the [diffusers repository](https://github.com/) from my GitHub.

## Process Overview

The notebook performs the following steps:

1. **Create New Masks**: Generate masks for images in the `crops` folder.
2. **Extract Skin Tones**: Use the new masks to extract skin tones from images.
3. **Create Crop-Specific Prompts**: Create prompts for each crop based on the extracted skin tones.
4. **Prepare the Dataset**: Integrate images, masks, and prompts into a dataset.
5. **Train the ControlNet Model**: Use the dataset to train a ControlNet model.
6. **Evaluate the Model**: Load a test set and apply metrics like PSNR, SSIM, and FID to evaluate the model's performance.

## Installation

To install the necessary libraries, run the following commands:
```bash
pip install --upgrade torch torchvision transformers diffusers torchmetrics accelerate lightning matplotlib opencv-python tabulate
```

## Running the Notebook

Follow these steps within the notebook:

1. **Creating New Masks**:
   Use `create_masks.py` to generate new masks and save them in the `crops_masks` folder.
   ```bash
   !python create_masks.py
   !cp crops/*[!_mask].png crops_masks/
   ```

2. **Extracting Skin Tones**:
   This step computes average pixel intensities from the masked regions of the images.

3. **Clustering Skin Tones**:
   Cluster skin tones into four groups (Pale, Light, Brown, Dark) for use in the prompt generation.

4. **Dataset Creation**:
   Integrate the images, masks, and generated prompts into a Hugging Face Dataset.

5. **Training the Model**:
   Use the Accelerate package to configure and launch training:
   ```bash
   accelerate launch train_controlnet.py --pretrained_model_name_or_path=<path> --controlnet_model_name_or_path=<path> --output_dir=<path> --dataset_name=skin_dataset --learning_rate=1e-5 --train_batch_size=8 --num_train_epochs=5
   ```

## Important Notes

- **Error Handling**: This notebook does not run in a Docker container. You may encounter errors. If so, follow the error messages and make the necessary adjustments.
- **GPU Support**: The notebook is designed to run on a machine with GPU support.

## Evaluation

Once training is complete, the model can be evaluated using PSNR, SSIM, and FID metrics. Example output for evaluation:

```bash
Avg PSNR: 27.19
Avg SSIM: 0.67
FID: 69.38
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

