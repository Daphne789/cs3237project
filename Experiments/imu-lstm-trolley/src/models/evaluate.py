# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This process will build on the existing work done by your friend, so ensure you have a clear understanding of their contributions and any existing code or models.

# ### Step 1: Understand Existing Work
# 1. **Review Documentation**: Go through any documentation or notes your friend has provided about their work.
# 2. **Analyze Code**: If your friend has shared code, review it to understand the architecture, data preprocessing, and any models they have implemented.
# 3. **Identify Gaps**: Determine what aspects of the project are incomplete or could be improved upon, such as data collection, model performance, or integration with the motor control system.

# ### Step 2: Data Collection and Preprocessing
# 1. **Gather IMU Data**: Ensure you have access to the IMU data your friend has collected. If not, set up a system to collect real-time IMU data (accelerometer, gyroscope, etc.).
# 2. **Data Cleaning**: Clean the data to remove any noise or irrelevant information. This may involve filtering or smoothing techniques.
# 3. **Feature Engineering**: Extract relevant features from the IMU data that may help in predicting actions (e.g., orientation, acceleration patterns).
# 4. **Labeling Data**: If not already done, label the data with the corresponding actions (e.g., move forward, turn left) that you want the LSTM to predict.

# ### Step 3: Model Development
# 1. **Choose LSTM Framework**: Decide on a framework for building your LSTM model (e.g., TensorFlow, Keras, PyTorch).
# 2. **Design the LSTM Architecture**: Based on your friend's work, design an LSTM architecture that suits your data and prediction task. Consider the number of layers, units, and dropout rates.
# 3. **Implement the Model**: Write the code to implement the LSTM model, ensuring it can take the preprocessed IMU data as input.

# ### Step 4: Training the Model
# 1. **Split Data**: Divide your dataset into training, validation, and test sets.
# 2. **Train the Model**: Train the LSTM model using the training set. Monitor the training process for overfitting or underfitting.
# 3. **Hyperparameter Tuning**: Experiment with different hyperparameters (learning rate, batch size, number of epochs) to optimize model performance.
# 4. **Evaluate Performance**: Use the validation set to evaluate the model's performance. Adjust the model as necessary based on the results.

# ### Step 5: Integration with Motor Control
# 1. **Define Control Logic**: Develop a control logic that translates the predicted actions from the LSTM model into motor commands for the trolley.
# 2. **Real-Time Implementation**: Implement the LSTM model in a real-time system where it can receive IMU data and output motor commands.
# 3. **Testing**: Test the integrated system in a controlled environment to ensure that the trolley responds correctly to the predicted actions.

# ### Step 6: Iteration and Improvement
# 1. **Collect Feedback**: Gather feedback from testing to identify any issues or areas for improvement.
# 2. **Refine the Model**: Based on feedback, refine the model, retrain it if necessary, and adjust the control logic.
# 3. **Document Changes**: Keep thorough documentation of any changes made to the model, control logic, and overall system.

# ### Step 7: Final Testing and Deployment
# 1. **Conduct Final Tests**: Perform comprehensive tests to ensure the system works reliably under various conditions.
# 2. **Prepare for Deployment**: If the system meets your requirements, prepare it for deployment in the intended environment.
# 3. **Monitor Performance**: After deployment, continuously monitor the system's performance and make adjustments as needed.

# ### Step 8: Collaboration and Knowledge Sharing
# 1. **Share Results**: Share your findings and improvements with your friend and any other collaborators.
# 2. **Collaborate on Future Work**: Discuss potential future work or enhancements that could be made to the project.

# By following these steps, you can effectively build upon your friend's work and develop a robust LSTM model for predicting actions based on real-time IMU data to control a trolley's motor movement.

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
from pathlib import Path
from src.data.loader import load_and_prepare
from src.models.lstm_model import ImprovedLSTMClassifier

def load_checkpoint(ckpt_path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt

def evaluate(ckpt_path, csv_path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(ckpt_path, device=device)
    classes = ckpt.get('label_classes')
    seq_length = ckpt.get('seq_length')
    scaler = load(Path(ckpt_path).parent/'scaler.joblib')
    le = load(Path(ckpt_path).parent/'label_encoder.joblib')
    X, y, _, _, _ = load_and_prepare(csv_path, seq_length=seq_length)
    model = ImprovedLSTMClassifier(input_size=X.shape[2], num_classes=len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    print(classification_report(y, preds, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))