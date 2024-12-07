import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

# Paths to your dataset
train_dir = r"C:\Users\yangq\OneDrive - Milwaukee School of Engineering\Desktop\dataset\images\train"
val_dir = r"C:\Users\yangq\OneDrive - Milwaukee School of Engineering\Desktop\dataset\images\val"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(train_dataset.classes))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), "efficientnet_volume_classifier.pth")

import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

# Path to the test folder
test_folder = r"C:\Users\yangq\OneDrive - Milwaukee School of Engineering\Desktop\test_images"

# Load the trained model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model.load_state_dict(torch.load("efficientnet_volume_classifier.pth"))
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classification function
def classify_images(folder, model, transform):
    results = []
    class_names = ["0ml", "50ml", "100ml", "500ml", "1000ml"]  # Replace with your actual class names

    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)

        # Ensure the file is an image
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = outputs.max(1)
                class_name = class_names[predicted.item()]
                results.append((img_file, class_name))
                print(f"Image {img_file} classified as: {class_name}")
        else:
            print(f"Skipping non-image file: {img_file}")

    return results

# Run classification
results = classify_images(test_folder, model, transform)

# Save results to file
output_file = r"C:\Users\yangq\OneDrive - Milwaukee School of Engineering\Desktop\classification_results.txt"
with open(output_file, 'w') as f:
    for img_file, class_name in results:
        f.write(f"Image {img_file} classified as: {class_name}\n")

