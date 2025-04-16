

# 6. Linear Probing Evaluation (as a separate function)
def linear_probing(model, pretrain_epochs, image_size=224, batch_size=64):
    """
    Performs linear probing evaluation on the pre-trained model.

    Args:
        model: The pre-trained model.
        pretrain_epochs: The number of epochs the model was pre-trained for.
        image_size: size of the images
        batch_size: the batch size
    Returns:
        The linear probing accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # 1. Freeze the pre-trained weights
    for param in model.parameters():
        param.requires_grad = False

    # 2. Create a new linear layer for classification (the "probe")
    embed_dim = model.embed_dim #  Get the embedding dimension from the model.
    num_classes = 1000 # ImageNet has 1000 classes
    linear_probe = nn.Linear(embed_dim, num_classes).to(device)
    optimizer_probe = optim.Adam(linear_probe.parameters(), lr=1e-3)
    criterion_probe = nn.CrossEntropyLoss()

    # 3. Load the ImageNet training and validation sets (or a subset)
    train_loader_probe = load_imagenet(batch_size, image_size=image_size, use_subset=True, subset_size=50000) # use a subset for probing
    val_loader_probe = load_imagenet(batch_size, image_size=image_size, use_subset=True, subset_size=10000)

    if train_loader_probe is None or val_loader_probe is None:
        return 0.0  # Handle the case where the dataset isn't found

    # 4. Train the linear probe
    print("Starting Linear Probing...")
    for epoch_probe in range(100):  # Example number of epochs for probing
        linear_probe.train()
        for batch_idx_probe, (images_probe, labels_probe) in enumerate(train_loader_probe):
            images_probe = images_probe.to(device)
            labels_probe = labels_probe.to(device)

            with torch.no_grad():
                # Get features from the frozen pre-trained model.  Use the CLS token.
                features_probe = model(images_probe)  # (B, N+1, D)
                features_probe = features_probe[:, 0, :]  # (B, D)

            # Forward pass through the linear probe
            outputs_probe = linear_probe(features_probe)  # (B, num_classes)
            loss_probe = criterion_probe(outputs_probe, labels_probe)

            # Backward pass and optimization for the probe ONLY
            optimizer_probe.zero_grad()
            loss_probe.backward()
            optimizer_probe.step()

            if batch_idx_probe % 100 == 0:
                print(f"Probe Epoch: {epoch_probe+1}, Batch: {batch_idx_probe}/{len(train_loader_probe)}, Loss: {loss_probe.item():.4f}")

        # 5. Evaluate the linear probe on the validation set
        linear_probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images_val, labels_val in val_loader_probe:
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)
                features_val = model(images_val)
                features_val = features_val[:, 0, :]
                outputs_val = linear_probe(features_val)
                _, predicted = torch.max(outputs_val.data, 1)
                total += labels_val.size(0)
                correct += (predicted == labels_val).sum().item()

        accuracy = correct / total
        print(f"Epoch {pretrain_epochs} Linear Probing Accuracy: {accuracy:.4f}")
        return accuracy
