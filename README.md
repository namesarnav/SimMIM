# SimMIM Implementation in PyTorch

This repository contains a PyTorch implementation of SimMIM (Simple Masked Image Modeling) for self-supervised visual representation learning.  This implementation is designed to help you understand and experiment with SimMIM.

## Code Description

This code provides a more robust PyTorch implementation of SimMIM, addressing some limitations of a simpler version.  It includes:

* **ViT Model:** Uses a pre-built ViT model from the `vit-pytorch` library for efficiency.
* **Masking:** Implements the random masking of image patches, a core component of SimMIM.
* **Dataset Loading:** Includes functions for loading ImageNet (or a subset) for pre-training.
* **Loss Function:** Calculates the masked mean squared error (MSE) for reconstruction.
* **Training:** Provides a training loop for the SimMIM pre-training process.
* **Linear Probing Evaluation:** Implements linear probing, the standard evaluation method for self-supervised visual representation learning.

**Important:** This implementation aims for clarity and completeness for educational purposes.  For state-of-the-art results, you might need further optimization and hyperparameter tuning.

## Requirements

* Python (3.7+)
* PyTorch (1.7 or later)
* Torchvision
* `vit-pytorch` library: `pip install vit-pytorch`

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Install the required packages:
    ```bash
    pip install torch torchvision vit-pytorch
    ```
3.  **Dataset:** Download the ImageNet dataset.  You will need to specify the path to your ImageNet data in the `load_imagenet` function within `simmim_robust.py`.  If you don't have the full ImageNet dataset, you can use the `use_subset=True` option for experimentation with a smaller dataset.

## Usage

1.  **Prepare your dataset:** Ensure ImageNet is downloaded and the path is correctly set in `simmim_robust.py`, or use the subset option.
2.  **Run the script:**
    ```bash
    python simmim_robust.py
    ```
    The script will first pre-train the SimMIM model and then perform linear probing evaluation.

## Code Structure

* `simmim_robust.py`:
    * `create_model()`:  Defines the ViT model.
    * `random_masking()`:  Implements the random patch masking.
    * `load_imagenet()`:  Loads the ImageNet dataset (or a subset).
    * `mse_loss()`:  Calculates the masked MSE loss.
    * `train_one_epoch()`:  Trains the model for one epoch.
    * `linear_probing()`:  Performs linear probing evaluation.
    * `main()`:  Main function to run the pre-training and evaluation.

## Hyperparameters

The following hyperparameters can be adjusted in the `main()` function of `simmim_robust.py`:

* `image_size`:  Input image size (default: 224).
* `patch_size`:  Size of the image patches (default: 16).
* `embed_dim`:  Embedding dimension of the ViT model (default: 768).
* `depth`:  Number of Transformer layers (default: 12).
* `num_heads`:  Number of attention heads (default: 12).
* `mlp_ratio`:  Ratio of MLP hidden layer size to embedding dimension (default: 4.0).
* `mask_ratio`:  Ratio of patches to mask (default: 0.75).
* `batch_size`:  Batch size (adjust based on your GPU memory).
* `epochs`:  Number of pre-training epochs (default: 100, but you should use a larger value for better results, e.g., 300+).
* `lr`:  Learning rate (default: 1e-4).

## Expected Output

The script will output the training loss during pre-training and the linear probing accuracy after pre-training is complete.  The exact accuracy will depend on your hardware, training time, and hyperparameters.  Remember that using a subset of ImageNet will result in lower accuracy than using the full dataset.

## Notes

* **Dataset:** Ensure that your ImageNet dataset is correctly downloaded and the path is set in the `load_imagenet` function.
* **Computational Resources:** Training a ViT model on ImageNet requires significant GPU resources.  Adjust the `batch_size` and `epochs` according to your hardware.  Consider using a smaller subset of ImageNet for initial experimentation.
* **Linear Probing:** Linear probing is crucial for evaluating the quality of the pre-trained representations.  The script performs this evaluation after pre-training.
* **Hyperparameter Tuning:** You may need to adjust the hyperparameters for optimal performance.

## Disclaimer

This implementation is for educational purposes.  It may not perfectly reproduce the results from the original SimMIM paper.

## Contributions

Contributions are welcome!  If you have any suggestions or improvements, feel free to submit a pull request.
