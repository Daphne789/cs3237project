# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This process will build on the existing work done by your friend, so make sure to review their work thoroughly before proceeding.

# ### Step 1: Review Existing Work
# 1. **Understand the Current Implementation**: Go through your friend's code, documentation, and any related materials to understand how they have set up the IMU data collection and any preliminary models they may have developed.
# 2. **Identify Data Sources**: Determine how the IMU data is being collected (e.g., sensors used, data format, sampling rate) and what preprocessing steps have already been implemented.

# ### Step 2: Define the Problem
# 1. **Specify the Objective**: Clearly define what actions you want to predict (e.g., forward, backward, turn left, turn right) based on the IMU data.
# 2. **Determine Output Format**: Decide how you will represent the actions (e.g., categorical labels, one-hot encoding).

# ### Step 3: Data Collection and Preprocessing
# 1. **Collect IMU Data**: If not already done, gather a sufficient amount of IMU data under various conditions to ensure the model can generalize well.
# 2. **Preprocess the Data**:
#    - Normalize or standardize the IMU data.
#    - Segment the data into sequences that can be fed into the LSTM (e.g., sliding window approach).
#    - Label the sequences with the corresponding actions.

# ### Step 4: Design the LSTM Model
# 1. **Choose a Framework**: Select a deep learning framework (e.g., TensorFlow, Keras, PyTorch) to implement the LSTM model.
# 2. **Model Architecture**:
#    - Define the architecture of the LSTM model (number of layers, number of units per layer, dropout rates, etc.).
#    - Consider adding additional layers (e.g., Dense layers) for output processing.

# ### Step 5: Train the Model
# 1. **Split the Data**: Divide the dataset into training, validation, and test sets.
# 2. **Compile the Model**: Choose an appropriate loss function (e.g., categorical cross-entropy) and optimizer (e.g., Adam).
# 3. **Train the Model**: Fit the model on the training data and validate it using the validation set. Monitor performance metrics (e.g., accuracy, loss) during training.

# ### Step 6: Evaluate the Model
# 1. **Test the Model**: Evaluate the trained model on the test set to assess its performance.
# 2. **Analyze Results**: Look for areas of improvement by analyzing confusion matrices, precision, recall, and F1 scores.

# ### Step 7: Real-Time Implementation
# 1. **Integrate with IMU Data Stream**: Set up a system to feed real-time IMU data into the trained LSTM model for predictions.
# 2. **Control Logic**: Develop control logic that translates model predictions into motor commands for the trolley.
# 3. **Test in Real-Time**: Conduct tests in a controlled environment to ensure the system responds correctly to real-time data.

# ### Step 8: Optimize and Iterate
# 1. **Fine-Tune the Model**: Based on real-time performance, consider retraining the model with additional data or adjusting hyperparameters.
# 2. **Implement Feedback Loops**: If possible, implement a feedback mechanism to improve predictions based on the trolley's performance.

# ### Step 9: Documentation and Reporting
# 1. **Document the Process**: Keep detailed records of your methodology, experiments, and results.
# 2. **Share Findings**: Prepare a report or presentation to share your findings and improvements with your friend and any other stakeholders.

# ### Step 10: Future Work
# 1. **Explore Advanced Techniques**: Consider exploring more advanced techniques such as attention mechanisms, transfer learning, or reinforcement learning for further improvements.
# 2. **Expand the Dataset**: Continue to collect more diverse data to improve model robustness.

# By following these steps, you can systematically build upon your friend's work and develop a robust LSTM model for predicting actions based on real-time IMU data for controlling a trolley's motor movement.

import numpy as np
import joblib
from pathlib import Path

# simple mapping from action label to motor commands
# motor_cmd: dict with left_speed,right_speed (range -1..1) and turn_angle (deg)
ACTION_TO_MOTOR = {
    'straight': {'left': 0.6, 'right': 0.6, 'angle': 0.0},
    'left':     {'left': 0.35, 'right': 0.6, 'angle': -25.0},
    'right':    {'left': 0.6, 'right': 0.35, 'angle': 25.0},
    'jump':     {'left': 0.0, 'right': 0.0, 'angle': 0.0},
    'default':  {'left': 0.0, 'right': 0.0, 'angle': 0.0}
}

def map_action_to_motor(action_label):
    return ACTION_TO_MOTOR.get(action_label, ACTION_TO_MOTOR['default'])

def accel_based_speed_scale(frames, gravity=9.8, min_scale=0.35, max_scale=1.2):
    """
    Compute a scale factor from accelerometer frames.
    frames: numpy array Nx6, columns [gx,gy,gz,ax,ay,az]
    Use last accel vector magnitude deviation from gravity as dynamic signal.
    """
    if frames is None or len(frames) == 0:
        return 1.0
    # accel columns are last three
    a = frames[-1, 3:6]
    norm = float(np.linalg.norm(a))
    dyn = max(0.0, abs(norm - gravity))  # small dynamic movement -> small dyn
    # scale dyn into [min_scale, max_scale] (tunable)
    # use simple linear mapping up to dyn_max ~ 6 m/s^2
    dyn_max = 6.0
    scale = min_scale + (max_scale - min_scale) * min(dyn / dyn_max, 1.0)
    return float(scale)

def smooth_motor(prev, new, alpha=0.4):
    """
    Exponential smoothing between prev and new dictionaries with numeric values.
    alpha: 0..1 weight for new (higher -> less smoothing)
    Returns dict same keys.
    """
    if prev is None:
        return new.copy()
    out = {}
    for k, v in new.items():
        pv = prev.get(k, 0.0)
        try:
            out[k] = float(alpha * v + (1.0 - alpha) * pv)
        except Exception:
            out[k] = v
    return out

def load_scaler_and_encoder(model_dir):
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir/'scaler.joblib')
    le = joblib.load(model_dir/'label_encoder.joblib')
    return scaler, le