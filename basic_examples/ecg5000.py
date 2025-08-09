import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class ECG_CNN(nn.Module):
    """
    CNN model for ECG5000 time series classification
    """
    def __init__(self, input_length=140, num_classes=5):
        super(ECG_CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.sequential1 = nn.Sequential(
            self.conv1,self.bn1,self.relu1,self.pool1,self.dropout1)
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        self.sequential2 = nn.Sequential(
            self.conv2, self.bn2, self.relu2, self.pool2, self.dropout2)
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.4)
        self.sequential3 = nn.Sequential(
            self.conv3, self.bn3, self.relu3, self.pool3, self.dropout3)
        # Calculate the size after convolutions
        # input_length -> input_length/2 -> input_length/4 -> input_length/8
        conv_output_size = 128 * (input_length // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.sequential1(x)
        x = self.sequential2(x)
        x = self.sequential3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def load_ecg5000_data():
    """
    Load and preprocess ECG5000 dataset
    """
    print("Loading ECG5000 dataset...")
    
    # Fetch the dataset
    ecg = fetch_openml(name='ECG5000', version=1, as_frame=False, parser="auto")
    
    # Extract features and labels
    X = ecg.data.astype(np.float32)
    y = ecg.target
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for CNN (batch_size, channels, sequence_length)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, le

def create_data_loaders(X, y, test_size=0.2, batch_size=32):
    """
    Create train and test data loaders
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    """
    Train the CNN model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Evaluation phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        test_acc = 100 * correct_test / total_test
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, label_encoder):
    """
    Evaluate the model and print detailed results
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    return accuracy

def plot_training_history(train_losses, train_accuracies, test_accuracies):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ecg5000_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the ECG5000 CNN training
    """
    # Load data
    X, y, label_encoder = load_ecg5000_data()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X, y, batch_size=64)
    
    # Initialize model
    input_length = X.shape[2]  # sequence length
    num_classes = len(np.unique(y))
    model = ECG_CNN(input_length=input_length, num_classes=num_classes)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, num_epochs=50, learning_rate=0.001
    )
    
    # Evaluate model
    final_accuracy = evaluate_model(model, test_loader, label_encoder)
    
    # Plot training history
    #plot_training_history(train_losses, train_accuracies, test_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'savez/ecg5000_cnn_model.pth')
    print(f"\nModel saved as 'ecg5000_cnn_model.pth'")
    
    return model, final_accuracy

def run():
        X, y, label_encoder = load_ecg5000_data()
        input_length = X.shape[2]  # sequence length
        num_classes = len(np.unique(y))
        model = ECG_CNN(input_length=input_length, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('savez/ecg5000_cnn_model.pth'))
        _, test_loader = create_data_loaders(X, y, batch_size=64)
        final_accuracy = evaluate_model(model, test_loader, label_encoder)
        print(f"Final Test Accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()