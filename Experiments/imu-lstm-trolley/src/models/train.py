# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This process assumes you have some existing work done by your friend that you can build upon.

# ### Step 1: Understand Existing Work
# 1. **Review Documentation**: Go through any documentation, code, or notes provided by your friend to understand their approach, data collection methods, and any models they have implemented.
# 2. **Analyze Data**: Examine the IMU data they have collected. Understand the features (e.g., accelerometer, gyroscope readings) and the target actions (e.g., move forward, turn left).
# 3. **Identify Gaps**: Determine what aspects of the project need improvement or further development. This could include data preprocessing, model architecture, or real-time implementation.

# ### Step 2: Data Preparation
# 1. **Collect Additional Data**: If necessary, gather more IMU data to enhance the dataset. Ensure that the data covers a wide range of actions and scenarios.
# 2. **Preprocess Data**:
#    - **Normalization**: Scale the IMU readings to a suitable range (e.g., 0 to 1).
#    - **Segmentation**: Divide the continuous IMU data into sequences that can be fed into the LSTM model. Each sequence should correspond to a specific action.
#    - **Labeling**: Ensure that each sequence is labeled with the corresponding action.

# ### Step 3: Model Development
# 1. **Choose a Framework**: Select a deep learning framework (e.g., TensorFlow, Keras, PyTorch) to implement the LSTM model.
# 2. **Design the LSTM Model**:
#    - **Input Layer**: Define the input shape based on the number of features and sequence length.
#    - **LSTM Layers**: Add one or more LSTM layers. Experiment with the number of units and layers to find the best configuration.
#    - **Dense Layer**: Add a dense layer for output, with the number of neurons corresponding to the number of actions.
#    - **Activation Function**: Use an appropriate activation function (e.g., softmax for multi-class classification).
# 3. **Compile the Model**: Choose a loss function (e.g., categorical cross-entropy) and an optimizer (e.g., Adam).

# ### Step 4: Model Training
# 1. **Split Data**: Divide the dataset into training, validation, and test sets.
# 2. **Train the Model**: Fit the model on the training data and validate it using the validation set. Monitor performance metrics (e.g., accuracy, loss).
# 3. **Hyperparameter Tuning**: Experiment with different hyperparameters (e.g., learning rate, batch size) to improve model performance.

# ### Step 5: Model Evaluation
# 1. **Test the Model**: Evaluate the model on the test set to assess its performance.
# 2. **Analyze Results**: Review the model's predictions and identify any patterns of misclassification. Use confusion matrices or classification reports for detailed insights.

# ### Step 6: Real-Time Implementation
# 1. **Integrate with IMU**: Set up a system to collect real-time IMU data from the sensors.
# 2. **Preprocess Real-Time Data**: Implement the same preprocessing steps used during training on the incoming real-time data.
# 3. **Make Predictions**: Use the trained LSTM model to predict actions based on the preprocessed real-time IMU data.
# 4. **Control Motor Movement**: Develop a control system that translates the predicted actions into motor commands for the trolley.

# ### Step 7: Testing and Iteration
# 1. **Test in Real Conditions**: Conduct tests in real-world scenarios to evaluate the performance of the system.
# 2. **Iterate**: Based on the results, refine the model, data collection, or control logic as needed. This may involve retraining the model with new data or adjusting the control algorithms.

# ### Step 8: Documentation and Reporting
# 1. **Document the Process**: Keep detailed records of your methodology, experiments, and findings.
# 2. **Share Results**: Prepare a report or presentation to share your results with your friend and any other stakeholders.

# ### Step 9: Future Work
# 1. **Explore Improvements**: Consider potential enhancements, such as using more advanced architectures (e.g., GRU, CNN-LSTM), incorporating additional sensors, or implementing reinforcement learning for better control.

# By following these steps, you can effectively build upon your friend's work and develop a robust LSTM model for predicting actions based on real-time IMU data to control a trolley's motor movement.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from joblib import dump, load
import matplotlib.pyplot as plt

from src.data.loader import load_and_prepare
from src.models.lstm_model import EnhancedLSTM

class IMUDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.01):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = x + np.random.normal(scale=self.noise_std, size=x.shape).astype(np.float32)
        return torch.from_numpy(x).float(), torch.tensor(int(self.y[idx]), dtype=torch.long)

def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    # inverse frequency
    weights = {c: 1.0/counts[i] for i,c in enumerate(classes)}
    sample_weights = np.array([weights[int(label)] for label in y], dtype=np.float32)
    return sample_weights, torch.tensor([weights[c] for c in classes], dtype=torch.float32)

def train(csv_path, save_dir, epochs=40, batch_size=32, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y, scaler, le, seq_length, _ = load_and_prepare(csv_path, save_assets_dir=str(save_dir))
    # split into train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

    # compute sample weights for balanced sampling
    sample_weights, class_weights = compute_class_weights(y_train)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = IMUDataset(X_train, y_train, augment=True, noise_std=0.02)
    val_ds = IMUDataset(X_val, y_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(le.classes_)
    model = EnhancedLSTM(input_size=X.shape[2], hidden_size=192, num_layers=2,
                         num_classes=num_classes, dropout=0.4, bidirectional=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)

    best_val = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_path = save_dir/'imu_lstm_model_best.pth'

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0; correct = 0; total = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights.to(device))
            loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss/total; train_acc = correct/total

        # validation
        model.eval()
        val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits, _ = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
        val_loss = val_loss/val_total; val_acc = val_correct/val_total

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        scheduler.step(val_loss)
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                'model_state_dict': model.state_dict(),
                'label_classes': list(le.classes_),
                'seq_length': int(seq_length),
                'val_acc': float(best_val)
            }
            torch.save(ckpt, best_path)

    dump(le, save_dir/'label_encoder.joblib')
    print("Training finished. Best val acc:", best_val)

    # plot curves
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(history['train_acc'], label='train_acc'); ax[0].plot(history['val_acc'], label='val_acc')
    ax[0].legend(); ax[0].grid(True); ax[0].set_title('Accuracy')
    ax[1].plot(history['train_loss'], label='train_loss'); ax[1].plot(history['val_loss'], label='val_loss')
    ax[1].legend(); ax[1].grid(True); ax[1].set_title('Loss')
    plt.tight_layout(); plt.savefig(save_dir/'training_curves.png'); plt.close(fig)

    # save weights-only checkpoint for compatibility
    if best_path.exists():
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False) if hasattr(torch.load, '__call__') else torch.load(best_path, map_location='cpu')
        torch.save({'model_state_dict': ckpt['model_state_dict']}, save_dir/'imu_lstm_model_weights_only.pth')

    # final test eval
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False) if hasattr(torch.load, '__call__') else torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            logits, _ = model(X_test_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        print("\n=== Test set evaluation ===")
        print(classification_report(y_test, preds, target_names=le.classes_))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, preds))
    else:
        print("No best checkpoint found for final evaluation.")
    return best_path