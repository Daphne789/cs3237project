# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This process assumes you have some existing work done by your friend, which you will build upon.

# ### Step 1: Understand Existing Work
# 1. **Review Documentation**: Go through any documentation, code, or notes your friend has provided. Understand the architecture, data preprocessing, and any models they have implemented.
# 2. **Analyze Data**: Examine the IMU data your friend has collected. Understand its structure, features, and how it relates to the trolley's motor actions.

# ### Step 2: Define the Problem
# 1. **Specify Objectives**: Clearly define what actions you want to predict (e.g., accelerate, decelerate, turn).
# 2. **Determine Output Format**: Decide how you will represent the actions (e.g., categorical labels, continuous values).

# ### Step 3: Data Preparation
# 1. **Collect Data**: If necessary, gather additional IMU data to augment what your friend has.
# 2. **Preprocess Data**:
#    - **Normalization**: Scale the IMU data to a suitable range (e.g., 0 to 1).
#    - **Segmentation**: Create sequences of data points that correspond to time windows for LSTM input.
#    - **Labeling**: Ensure each sequence is labeled with the corresponding action.

# ### Step 4: Model Development
# 1. **Choose Framework**: Select a deep learning framework (e.g., TensorFlow, PyTorch) that you are comfortable with.
# 2. **Design LSTM Model**:
#    - **Input Layer**: Define the input shape based on your data.
#    - **LSTM Layers**: Add one or more LSTM layers. Experiment with the number of units and layers.
#    - **Dense Layer**: Add a dense layer for output, with activation functions suitable for your output format (e.g., softmax for classification).
# 3. **Compile Model**: Choose an appropriate loss function (e.g., categorical cross-entropy for classification) and optimizer (e.g., Adam).

# ### Step 5: Training the Model
# 1. **Split Data**: Divide your dataset into training, validation, and test sets.
# 2. **Train the Model**: Fit the model on the training data, using the validation set to monitor performance.
# 3. **Hyperparameter Tuning**: Experiment with different hyperparameters (learning rate, batch size, number of epochs) to improve performance.

# ### Step 6: Evaluate the Model
# 1. **Test the Model**: Evaluate the model on the test set to assess its performance.
# 2. **Metrics**: Use appropriate metrics (accuracy, precision, recall, F1-score) to quantify performance.

# ### Step 7: Real-Time Implementation
# 1. **Integrate with Hardware**: Connect the model to the trolley's control system. Ensure you can receive real-time IMU data.
# 2. **Real-Time Prediction**: Implement a loop that continuously reads IMU data, preprocesses it, and feeds it into the LSTM model for action prediction.
# 3. **Control Logic**: Develop logic to translate model predictions into motor commands.

# ### Step 8: Testing and Iteration
# 1. **Field Testing**: Test the trolley in real-world scenarios to evaluate performance.
# 2. **Iterate**: Based on the results, refine the model, data preprocessing, or control logic as needed.

# ### Step 9: Documentation and Reporting
# 1. **Document Your Work**: Keep detailed notes on your process, findings, and any changes made to the original work.
# 2. **Prepare a Report**: Summarize your methodology, results, and any recommendations for future work.

# ### Step 10: Collaboration and Feedback
# 1. **Share Findings**: Discuss your results with your friend and seek feedback.
# 2. **Collaborate on Improvements**: Work together to identify areas for further enhancement or research.

# By following these steps, you can systematically build upon your friend's work and develop a robust LSTM model for predicting actions based on real-time IMU data for controlling the trolley's motor movement.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

# base columns from your CSV
BASE_FEATURES = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']

def _add_features(arr):
    # arr: (T,6)
    gx = arr[:,0]; gy = arr[:,1]; gz = arr[:,2]
    ax = arr[:,3]; ay = arr[:,4]; az = arr[:,5]
    accel_mag = np.linalg.norm(arr[:,3:6], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(arr[:,0:3], axis=1, keepdims=True)
    # deltas (first differences), pad with zeros
    delta = np.vstack([np.zeros((1,6)), arr[1:,:] - arr[:-1,:]])
    delta_mag = np.linalg.norm(delta[:,3:6], axis=1, keepdims=True)
    # stack: original + mags + delta mags
    out = np.hstack([arr, accel_mag, gyro_mag, delta_mag])
    return out.astype(np.float32)

def load_and_prepare(csv_path, seq_length=None, percentile=95, save_assets_dir=None, regression_targets=None):
    """
    Load CSV and build sequences.
    - regression_targets: list of column names for regression (optional)
    Returns: X (N,seq,feat), y (N,) or None if regression, scaler, label_encoder (may be None), seq_length, reg_targets (NxK) or None
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=BASE_FEATURES + ['action_id','action_label'])
    sequences = []
    labels = []
    reg_list = []
    for aid in df['action_id'].unique():
        g = df[df['action_id']==aid].sort_values('timestamp')
        seq = g[BASE_FEATURES].values.astype(np.float32)
        if seq.shape[0] == 0:
            continue
        sequences.append(seq)
        labels.append(g['action_label'].mode().iloc[0])
        if regression_targets:
            # take per-sequence mean of target columns (or you can store per-frame)
            vals = g[regression_targets].mean(axis=0).values.astype(np.float32)
            reg_list.append(vals)
    if len(sequences) == 0:
        raise RuntimeError("No sequences found in CSV")

    lengths = [len(s) for s in sequences]
    if seq_length is None:
        seq_length = int(np.percentile(lengths, percentile))
        seq_length = max(10, seq_length)

    # feature engineering transform
    sequences_feat = [_add_features(s) for s in sequences]
    all_rows = np.vstack(sequences_feat)
    scaler = StandardScaler().fit(all_rows)
    sequences_scaled = [scaler.transform(s) for s in sequences_feat]

    # pad/truncate
    feat_dim = sequences_scaled[0].shape[1]
    def pad_trunc(x, L):
        if len(x) >= L:
            return x[:L]
        pad = np.zeros((L - len(x), feat_dim), dtype=np.float32)
        return np.vstack([x, pad])
    X = np.stack([pad_trunc(s, seq_length) for s in sequences_scaled])

    le = LabelEncoder()
    y = le.fit_transform(labels)

    reg_targets_arr = None
    if regression_targets:
        reg_targets_arr = np.stack(reg_list)

    if save_assets_dir:
        Path(save_assets_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, Path(save_assets_dir)/'scaler.joblib')
        joblib.dump(le, Path(save_assets_dir)/'label_encoder.joblib')

    return X, y, scaler, le, seq_length, reg_targets_arr