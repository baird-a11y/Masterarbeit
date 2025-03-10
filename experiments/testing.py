import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# Transformations definieren
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((375, 1242)),  # Einheitliche Größe
    transforms.ToTensor()
])

# Dataset-Klasse aktualisieren
class KITTISemanticDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, target_size=(375, 1242)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir)) if label_dir else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path))

        # Resize Bild
        image = resize(torch.tensor(image).permute(2, 0, 1).float() / 255.0, self.target_size)

        if self.labels:
            label_path = os.path.join(self.label_dir, self.labels[idx])
            label = np.array(Image.open(label_path))
            label = resize(torch.tensor(label).unsqueeze(0).long(), self.target_size, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
            return image, label
        return image

# Ordner für Trainings- und Testdaten definieren
train_image_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
train_label_dir = "S:/Masterarbeit/Datensatz/Training/semantic"
test_image_dir = "S:/Masterarbeit/Datensatz/Training/image_2"
test_label_dir = "S:/Masterarbeit/Datensatz/Training/semantic"

# Dataset und DataLoader für Trainingsdaten
train_dataset = KITTISemanticDataset(train_image_dir, train_label_dir, transform=resize_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Dataset und DataLoader für Testdaten
test_dataset = KITTISemanticDataset(test_image_dir, test_label_dir, transform=resize_transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# U-Net Architektur
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=35):  # Multi-Klassen-Segmentierung
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.double_conv(in_channels, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)
        self.encoder4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.double_conv(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.double_conv(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.double_conv(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        print(f"enc1 dimensions: {enc1.shape}")
        enc2 = self.encoder2(nn.MaxPool2d(2, padding=1)(enc1))
        print(f"enc2 dimensions: {enc2.shape}")
        enc3 = self.encoder3(nn.MaxPool2d(2, padding=1)(enc2))
        print(f"enc3 dimensions: {enc3.shape}")
        enc4 = self.encoder4(nn.MaxPool2d(2, padding=1)(enc3))
        print(f"enc4 dimensions: {enc4.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2, padding=1)(enc4))
        print(f"bottleneck dimensions: {bottleneck.shape}")

        # Decoder
        dec4 = self.upconv4(bottleneck)
        print(f"dec4 dimensions: {dec4.shape}")
        dec4 = torch.cat((dec4[:, :, :enc4.size(2), :enc4.size(3)], enc4), dim=1)
        print(f"dec4 dimensions after concatenation: {dec4.shape}")
        dec4 = self.decoder4(dec4)
        print(f"dec4 dimensions after decoder: {dec4.shape}")
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3[:, :, :enc3.size(2), :enc3.size(3)], enc3), dim=1)
        dec3 = self.decoder3(dec3)
        print(f"dec3 dimensions: {dec3.shape}")
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2[:, :, :enc2.size(2), :enc2.size(3)], enc2), dim=1)
        dec2 = self.decoder2(dec2)
        print(f"dec2 dimensions: {dec2.shape}")
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1[:, :, :enc1.size(2), :enc1.size(3)], enc1), dim=1)
        dec1 = self.decoder1(dec1)
        print(f"dec1 dimensions: {dec1.shape}")
        # Output
        out = self.output(dec1)
        return out

# Trainingsparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=35).to(device)
epochs = 1
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()  # Multi-Klassen-Verlust
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Debugging-Statements
print(f"Trainingsdaten: {len(train_dataset)} Bilder")
print(f"Testdaten: {len(test_dataset)} Bilder")
print(f"Device: {device}")

# Trainingsschleife
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_dataloader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Vorhersagen
        outputs = model(images)

        # Verlust berechnen
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verlust summieren
        epoch_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataloader):.4f}")

# Evaluation auf Testdaten
model.eval()
ious = []
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).argmax(dim=1)

        # IoU berechnen
        for true, pred in zip(labels.cpu().numpy(), outputs.cpu().numpy()):
            ious.append(jaccard_score(true.flatten(), pred.flatten(), average="weighted"))
print(f"Durchschnittliche IoU auf Testdaten: {np.mean(ious):.4f}")

# Modell speichern
torch.save(model.state_dict(), "unet_model.pth")


def visualize_and_save_predictions(model, test_dataloader, device, save_dir, num_samples=5):
    """
    Visualisiert Testbilder, Ground Truth Labels und Modellvorhersagen
    und speichert die Ergebnisse im angegebenen Ordner.
    """
    model.eval()  # Modell in den Evaluierungsmodus versetzen
    os.makedirs(save_dir, exist_ok=True)  # Sicherstellen, dass der Ordner existiert
    samples = 0

    # Testbilder durchgehen
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            predictions = model(images).argmax(dim=1)  # Vorhersagen (Klassen) erzeugen

        # Für jede Bild-Vorhersage-Kombination speichern
        for i in range(len(images)):
            if samples >= num_samples:
                return

            # Visualisierung erstellen
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Originalbild
            axs[0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axs[0].set_title("Original")
            axs[0].axis('off')
            
            # Ground Truth Label
            axs[1].imshow(labels[i].cpu().numpy(), cmap="viridis")
            axs[1].set_title("Lösung (Ground Truth)")
            axs[1].axis('off')
            
            # Vorhersage
            axs[2].imshow(predictions[i].cpu().numpy(), cmap="viridis")
            axs[2].set_title("Vorhersage")
            axs[2].axis('off')
            
            plt.tight_layout()

            # Datei speichern
            save_path = os.path.join(save_dir, f"sample_{samples+1}.png")
            plt.savefig(save_path)
            plt.close(fig)  # Speicher freigeben

            samples += 1

# Pfad zum Speichern der Visualisierungen
save_dir = r"G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/results"

# Visualisierung der Vorhersagen durchführen und speichern
visualize_and_save_predictions(model, test_dataloader, device, save_dir, num_samples=5)