# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This plan assumes you have some existing work done by your friend, which you can build upon.

# ### Step 1: Understand Existing Work
# 1. **Review Documentation**: Go through any documentation, code, and results your friend has provided. Understand the architecture, data preprocessing, and the model they used.
# 2. **Identify Key Components**: Note the key components of their work, such as data collection methods, feature extraction, model architecture, and evaluation metrics.

# ### Step 2: Define the Problem
# 1. **Clarify Objectives**: Define what specific actions you want to predict (e.g., stop, move forward, turn).
# 2. **Determine Output Format**: Decide how you will represent the actions (e.g., categorical labels, continuous values).

# ### Step 3: Data Collection and Preprocessing
# 1. **Gather IMU Data**: Ensure you have access to the IMU data your friend used or collect new data if necessary.
# 2. **Data Cleaning**: Clean the data to remove noise and handle missing values.
# 3. **Feature Engineering**: Extract relevant features from the IMU data (e.g., acceleration, angular velocity) that may help in predicting actions.
# 4. **Normalization**: Normalize the data to ensure that all features contribute equally to the model training.

# ### Step 4: Prepare Data for LSTM
# 1. **Sequence Creation**: Convert the IMU data into sequences suitable for LSTM input. This typically involves creating overlapping windows of time-series data.
# 2. **Labeling**: Ensure each sequence is labeled with the corresponding action that should be predicted.
# 3. **Train-Test Split**: Split the data into training, validation, and test sets.

# ### Step 5: Model Development
# 1. **Select LSTM Framework**: Choose a deep learning framework (e.g., TensorFlow, Keras, PyTorch) to implement the LSTM model.
# 2. **Model Architecture**: Design the LSTM architecture. Start with a simple model and gradually increase complexity (e.g., adding layers, dropout for regularization).
# 3. **Compile the Model**: Choose an appropriate loss function and optimizer. For classification, you might use categorical cross-entropy; for regression, mean squared error.

# ### Step 6: Training the Model
# 1. **Train the Model**: Fit the model on the training data while monitoring performance on the validation set.
# 2. **Hyperparameter Tuning**: Experiment with different hyperparameters (e.g., learning rate, batch size, number of epochs) to improve performance.
# 3. **Early Stopping**: Implement early stopping to prevent overfitting.

# ### Step 7: Evaluate the Model
# 1. **Test the Model**: Evaluate the model on the test set to assess its performance.
# 2. **Metrics**: Use appropriate metrics (e.g., accuracy, precision, recall, F1-score) to quantify the model's performance.
# 3. **Confusion Matrix**: Analyze the confusion matrix to understand where the model is making errors.

# ### Step 8: Integration with Motor Control
# 1. **Real-Time Data Handling**: Set up a system to handle real-time IMU data input.
# 2. **Action Prediction**: Implement the model to predict actions based on incoming IMU data.
# 3. **Motor Control Interface**: Develop an interface to control the trolley's motors based on the predicted actions.

# ### Step 9: Testing and Validation
# 1. **Simulated Environment**: Test the system in a controlled environment to validate the predictions and motor responses.
# 2. **Iterate**: Based on the results, iterate on the model and control logic to improve performance.

# ### Step 10: Documentation and Reporting
# 1. **Document the Process**: Keep detailed records of your methodology, experiments, and results.
# 2. **Share Findings**: Prepare a report or presentation to share your findings with your friend and any other stakeholders.

# ### Step 11: Future Improvements
# 1. **Feedback Loop**: Implement a feedback mechanism to continuously improve the model based on real-world performance.
# 2. **Explore Advanced Techniques**: Consider exploring more advanced techniques such as transfer learning, attention mechanisms, or hybrid models if needed.

# By following these steps, you can systematically build upon your friend's work and develop a robust LSTM model for predicting actions based on real-time IMU data for controlling a trolley's motor movement.

import time
import requests
import pandas as pd
from pathlib import Path

# def stream_csv_actions(csv_path, server_url='http://127.0.0.1:5000/predict', delay_per_frame=0.02):
def stream_csv_actions(csv_path, server_url='http://127.0.0.1:5000/predict?log=1', delay_per_frame=0.02):
    df = pd.read_csv(csv_path)
    FEATURE_COLS = ['gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z']
    for aid in df['action_id'].unique():
        g = df[df['action_id']==aid].sort_values('timestamp')
        frames = g[FEATURE_COLS].values.tolist()
        # send sliding windows (simulate live frames)
        buf = []
        for f in frames:
            buf.append([float(x) for x in f])
            payload = {'frames': buf}
            try:
                r = requests.post(server_url, json=payload, timeout=2.0)
                print("resp:", r.json())
            except Exception as e:
                print("server error:", e)
            time.sleep(delay_per_frame)

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else 'data/raw/all-combined_imu_data_randomized.csv'
    stream_csv_actions(path)