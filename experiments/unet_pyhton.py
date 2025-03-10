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

# Einheitliche Zielgröße für alle Bilder definieren
TARGET_SIZE = (352, 1216)  # Angepasst auf eine Größe, die durch 32 teilbar ist (für U-Net)

# Dataset-Klasse
class KITTISemanticDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, target_size=TARGET_SIZE):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        
        # Sicherstellen, dass die Ordner existieren
        assert os.path.exists(image_dir), f"Bildordner existiert nicht: {image_dir}"
        if label_dir:
            assert os.path.exists(label_dir), f"Labelordner existiert nicht: {label_dir}"
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        if label_dir:
            self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))])
            # Sicherstellen, dass gleiche Anzahl von Bildern und Labels vorhanden sind
            assert len(self.images) == len(self.labels), "Unterschiedliche Anzahl von Bildern und Labels"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Einheitliches Resize aller Bilder mit einem sauberen Transformationsprozess
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize(self.target_size)(image)
        
        if self.label_dir:
            label_path = os.path.join(self.label_dir, self.labels[idx])
            label = Image.open(label_path)
            # Resize für Label mit NEAREST Interpolation um Labelklassen zu erhalten
            label = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)(
                transforms.ToTensor()(label)
            )
            # Labels als Integer (0, 1, 2, ...) statt One-Hot
            label = label.squeeze(0).long()
            return image, label
        
        return image

# U-Net Architektur mit ordnungsgemäßem Padding
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=35):
        super(UNet, self).__init__()
        
        # Encoder (Kontraktion)
        self.encoder1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck (Brücke)
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder (Expansion) mit Skip-Connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(1024, 512)  # 1024 wegen Konkatenation
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(512, 256)   # 512 wegen Konkatenation
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(256, 128)   # 256 wegen Konkatenation
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(128, 64)    # 128 wegen Konkatenation
        
        # Ausgabeschicht
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder mit Skip-Connections
        x = self.upconv4(x)
        # Center-Crop für Skip-Connection
        diffY = enc4.size()[2] - x.size()[2]
        diffX = enc4.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([enc4, x], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        # Center-Crop für Skip-Connection
        diffY = enc3.size()[2] - x.size()[2]
        diffX = enc3.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([enc3, x], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        # Center-Crop für Skip-Connection
        diffY = enc2.size()[2] - x.size()[2]
        diffX = enc2.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([enc2, x], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        # Center-Crop für Skip-Connection
        diffY = enc1.size()[2] - x.size()[2]
        diffX = enc1.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([enc1, x], dim=1)
        x = self.decoder1(x)
        
        return self.output(x)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BatchNorm für bessere Stabilität
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# Hauptfunktion für Training und Evaluation
def train_and_evaluate_unet(train_image_dir, train_label_dir, 
                            val_image_dir=None, val_label_dir=None,
                            batch_size=4, epochs=10, learning_rate=0.001,
                            save_dir="results"):
    
    # Gerät für Training (GPU oder CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {device}")
    
    # Transformationen definieren
    train_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        # Optional: Datenerweiterung für bessere Generalisierung
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor()
    ])
    
    # Datasets und DataLoader erstellen
    train_dataset = KITTISemanticDataset(train_image_dir, train_label_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Validierungsdaten (kann gleich den Testdaten sein, wenn keine separaten Validierungsdaten vorhanden sind)
    if val_image_dir is None:
        val_image_dir = train_image_dir
    if val_label_dir is None:
        val_label_dir = train_label_dir
    
    val_dataset = KITTISemanticDataset(val_image_dir, val_label_dir, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Trainingsdaten: {len(train_dataset)} Bilder")
    print(f"Validierungsdaten: {len(val_dataset)} Bilder")
    
    # Modell, Verlustfunktion und Optimizer instanziieren
    model = UNet(in_channels=3, out_channels=35).to(device)  # 35 Klassen für KITTI
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lernratenplanung für bessere Konvergenz
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Trainingsverlauf aufzeichnen
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }
    
    # Trainingsschleife
    best_iou = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, labels in train_loop:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validierung
        model.eval()
        val_loss = 0.0
        ious = []
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Beste Klasse pro Pixel für IoU berechnen
                preds = outputs.argmax(dim=1)
                
                # IoU für jeden Batch berechnen
                for i in range(preds.size(0)):
                    pred = preds[i].cpu().numpy().flatten()
                    label = labels[i].cpu().numpy().flatten()
                    iou = jaccard_score(label, pred, average='weighted', zero_division=1)
                    ious.append(iou)
                
                val_loop.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_iou = np.mean(ious)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_iou)
        
        # Lernrate aktualisieren
        scheduler.step(avg_val_loss)
        
        # Modell speichern, wenn es das beste ist
        if avg_iou > best_iou:
            best_iou = avg_iou
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_best_model.pth"))
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, IoU = {avg_iou:.4f}")
    
    # Finales Modell speichern
    torch.save(model.state_dict(), os.path.join(save_dir, "unet_final_model.pth"))
    
    # Trainingsverlauf visualisieren
    plot_training_history(history, save_dir)
    
    # Einige Beispiele visualisieren
    visualize_predictions(model, val_dataloader, device, save_dir, num_samples=5)
    
    return model, history

# Funktion zum Visualisieren des Trainingsverlaufs
def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Loss-Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # IoU-Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Validation IoU Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

# Visualisierungsfunktion für Vorhersagen
def visualize_predictions(model, dataloader, device, save_dir, num_samples=5):
    """
    Visualisiert Eingabebilder, Ground Truth und Modellvorhersagen
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            
            # Vorhersagen generieren
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Mehrere Samples aus dem Batch visualisieren
            for i in range(min(len(images), num_samples - sample_count)):
                # Originalbild
                img = images[i].cpu().permute(1, 2, 0).numpy()
                
                # Ground Truth
                label = labels[i].cpu().numpy()
                
                # Modellvorhersage
                pred = preds[i].cpu().numpy()
                
                # Visualisierung
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(label, cmap='viridis')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred, cmap='viridis')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{sample_count+1}.png'))
                plt.close()
                
                sample_count += 1
                if sample_count >= num_samples:
                    break

if __name__ == "__main__":
    # Konfiguration
    train_image_dir = "S:/Masterarbeit/Datensatz/Training/image_2"  # Bitte anpassen
    train_label_dir = "S:/Masterarbeit/Datensatz/Training/semantic"  # Bitte anpassen
    test_image_dir = "S:/Masterarbeit/Datensatz/Training/image_2"  # Bitte anpassen 
    test_label_dir = "S:/Masterarbeit/Datensatz/Training/semantic"  # Bitte anpassen
    
    results_dir = "results"
    batch_size = 4
    epochs = 10
    
    # Training und Evaluation
    model, history = train_and_evaluate_unet(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        val_image_dir=test_image_dir,
        val_label_dir=test_label_dir,
        batch_size=batch_size,
        epochs=epochs,
        save_dir=results_dir
    )
    
    print(f"Ausführung abgeschlossen. Ergebnisse gespeichert in: {results_dir}")