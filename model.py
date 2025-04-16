import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import time
from vit_pytorch.simple_vit import SimpleViT  # Using a pre-built ViT implementation

# 1. Model Definition (Using a pre-built ViT)
def create_model(image_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
    """
    Creates a ViT model suitable for SimMIM.  Uses a pre-built ViT for simplicity.

    Args:
        image_size: Input image size.
        patch_size: Size of the image patches.
        embed_dim: Dimensionality of the token embeddings.
        depth: Number of Transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of the hidden layer size to the embedding dimension in the MLP.

    Returns:
        A PyTorch nn.Module representing the ViT model.
    """
    model = SimpleViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1000,  # We'll ignore this for pre-training, but it's required by the library
        dim=embed_dim,
        depth=depth,
        heads=num_heads,
        mlp_dim=int(embed_dim * mlp_ratio),
    )

    # Replace the classification head with a reconstruction head
    model.mlp_head = nn.Linear(embed_dim, patch_size * patch_size * 3)
    return model



# 3. Dataset Loading (ImageNet or a subset)
def load_imagenet(batch_size, image_size=224, use_subset=False, subset_size=10000):
    """
    Loads ImageNet or a subset of ImageNet.

    Args:
        batch_size: Batch size.
        image_size: Size to resize images to.
        use_subset: Whether to use a subset of the data.
        subset_size: The number of images to use in the subset.

    Returns:
        A PyTorch DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if use_subset:
        # Load the entire ImageNet validation set
        dataset = datasets.ImageNet(root='./data', split='val', transform=transform)

        # Create a random subset
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = torch.utils.data.Subset(dataset, indices)
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Using a subset of ImageNet with {subset_size} images.")
        return data_loader

    else:
        try:
            # This requires you to have the ImageNet dataset downloaded and the path set up correctly.
            dataset = datasets.ImageNet(root='./data', split='train', transform=transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            print("Using the full ImageNet training set.")
            return data_loader
        except:
            print("Error: ImageNet dataset not found. Please download and set up the path correctly.")
            print("If you want to run with a smaller dataset, use the use_subset=True option.")
            return None



# 4. Loss Function
def mse_loss(pred, target, mask):
    """
    Masked Mean Squared Error.

    Args:
        pred: (B, N, P*P*C)
        target: (B, N, P*P*C)
        mask: (B, N)

    Returns:
        The masked MSE loss.
    """
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # (B, N)
    loss = loss[mask].mean()
    return loss


# 5. Training Function
def train_one_epoch(model, optimizer, data_loader, device, epoch, mask_ratio):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        data_loader: The DataLoader.
        device: The device to train on (CPU or GPU).
        epoch: The current epoch number.
        mask_ratio: The ratio of patches to mask.

    Returns:
        The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (images, _) in enumerate(data_loader):
        if images is None:
            return 0 # Handle the case where the dataset isn't found

        images = images.to(device)
        batch_size = images.size(0)

        # Patchify
        patches = model.patchify(images)  # (B, N, P*P*C)
        # Mask
        masked_patches, mask = random_masking(patches, mask_ratio)  # (B, N, P*P*C), (B, N)

        # Forward pass
        predictions = model(patches, mask=mask)  # (B, N, P*P*C)

        # Loss
        loss = mse_loss(predictions, patches, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}] Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Time: {time.time() - start_time:.2f}s"
            )
            start_time = time.time()

    return total_loss / len(data_loader)





# 7. Main Function
def main():
    # Hyperparameters (Adjust as needed)
    image_size = 224
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    mask_ratio = 0.75
    batch_size = 64  # Adjust based on your GPU memory
    epochs = 100  # You'll want to train for longer, like 300+ for good results.
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader = load_imagenet(batch_size, image_size=image_size, use_subset=True, subset_size=10000) # Use subset for pretraining

    if train_loader is None:
        return  # Exit if dataset loading fails.

    # Model
    model = create_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr) #  Use AdamW

    # Train
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, mask_ratio)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    print("Pre-training finished! Starting Linear Probing Evaluation...")
    # Perform Linear Probing
    linear_probing_accuracy = linear_probing(model, epochs, image_size=image_size, batch_size=batch_size)
    print(f"Final Linear Probing Accuracy: {linear_probing_accuracy:.4f}")


if __name__ == "__main__":
    main()

