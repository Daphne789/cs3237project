# To achieve the goal of using Long Short-Term Memory (LSTM) networks to predict actions based on real-time Inertial Measurement Unit (IMU) data for controlling a trolley's motor movement, you can follow these step-by-step actions. This process will build on the existing work done by your friend, so make sure to review their work thoroughly before proceeding.

# ### Step 1: Review Existing Work
# 1. **Understand the Current Implementation**: Go through your friend's code, documentation, and any results they have achieved. Identify the data they used, the model architecture, and the performance metrics.
# 2. **Identify Gaps**: Determine what aspects of the project can be improved or expanded upon. This could include data quality, model performance, or additional features.

# ### Step 2: Data Collection and Preprocessing
# 1. **Gather IMU Data**: Ensure you have access to the IMU data your friend used. If necessary, collect additional data to enhance the dataset.
# 2. **Data Cleaning**: Remove any noise or irrelevant data points. Handle missing values appropriately.
# 3. **Feature Engineering**: Extract relevant features from the IMU data (e.g., acceleration, angular velocity) that may help in predicting motor actions.
# 4. **Normalization**: Normalize the data to ensure that all features contribute equally to the model training.

# ### Step 3: Define the Problem
# 1. **Specify the Output**: Clearly define what actions you want to predict (e.g., forward, backward, stop, turn).
# 2. **Determine Input Sequence Length**: Decide how many time steps of IMU data will be used as input for the LSTM model.

# ### Step 4: Model Development
# 1. **Choose a Framework**: Select a deep learning framework (e.g., TensorFlow, Keras, PyTorch) that you are comfortable with.
# 2. **Design the LSTM Model**: 
#    - Start with a simple LSTM architecture (e.g., one or two LSTM layers followed by dense layers).
#    - Experiment with hyperparameters such as the number of units, dropout rates, and activation functions.
# 3. **Compile the Model**: Choose an appropriate loss function (e.g., categorical cross-entropy for classification) and optimizer (e.g., Adam).

# ### Step 5: Model Training
# 1. **Split the Data**: Divide your dataset into training, validation, and test sets.
# 2. **Train the Model**: Fit the model on the training data while monitoring performance on the validation set.
# 3. **Hyperparameter Tuning**: Experiment with different hyperparameters and architectures to improve model performance.

# ### Step 6: Model Evaluation
# 1. **Evaluate Performance**: Use the test set to evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
# 2. **Analyze Results**: Visualize the model's predictions versus actual actions to identify any patterns or areas for improvement.

# ### Step 7: Real-Time Implementation
# 1. **Integrate with Hardware**: Connect the LSTM model to the trolley's control system. Ensure that the model can receive real-time IMU data.
# 2. **Implement Inference**: Write code to preprocess incoming IMU data, feed it into the trained LSTM model, and interpret the model's output to control the trolley's motors.
# 3. **Test in Real-Time**: Conduct tests in a controlled environment to ensure that the trolley responds correctly to the predicted actions.

# ### Step 8: Iteration and Improvement
# 1. **Collect Feedback**: Gather feedback from tests and make adjustments as necessary.
# 2. **Refine the Model**: Based on the performance in real-time scenarios, consider retraining the model with additional data or tweaking the architecture.
# 3. **Document Changes**: Keep thorough documentation of any changes made to the model, data processing, and results.

# ### Step 9: Finalize and Deploy
# 1. **Finalize the Model**: Once satisfied with the performance, finalize the model for deployment.
# 2. **Create User Documentation**: Write documentation for future users, explaining how to use the system and any maintenance required.
# 3. **Deploy the System**: Implement the system in the intended environment, ensuring all components work seamlessly together.

# ### Step 10: Continuous Monitoring and Maintenance
# 1. **Monitor Performance**: Continuously monitor the system's performance in real-time and make adjustments as needed.
# 2. **Update the Model**: Periodically retrain the model with new data to improve accuracy and adapt to changing conditions.

# By following these steps, you can effectively leverage LSTM networks to predict actions based on real-time IMU data, enhancing the control of the trolley's motor movement while building on your friend's foundational work.

import requests
import logging
import json
from typing import Tuple

LOG = logging.getLogger("motor_controller")

def send_motor_command(chassis_url: str, payload: dict, timeout: float = 1.0, retries: int = 2) -> Tuple[bool, str]:
    """
    POST motor command to chassis_url. Returns (success, response_text_or_error).
    On failure, does NOT block; caller may call safe_stop().
    """
    for attempt in range(retries):
        try:
            r = requests.post(chassis_url.rstrip("/") + "/motor", json=payload, timeout=timeout)
            r.raise_for_status()
            return True, r.text
        except Exception as e:
            LOG.warning("send_motor_command attempt %d failed: %s", attempt+1, e)
            last_err = str(e)
    return False, last_err

def safe_stop(chassis_url: str, timeout: float = 0.8) -> Tuple[bool, str]:
    """
    Send an emergency stop (zero speeds) to chassis.
    """
    try:
        payload = {"action":"emergency_stop", "motor":{"left":0.0,"right":0.0,"angle":0.0,"speed":0.0}}
        r = requests.post(chassis_url.rstrip("/") + "/motor", json=payload, timeout=timeout)
        r.raise_for_status()
        LOG.info("safe_stop sent")
        return True, r.text
    except Exception as e:
        LOG.error("safe_stop failed: %s", e)
        return False, str(e)

if __name__ == '__main__':
    # quick dry-run test (no chassis)
    print(send_motor_command(None, {'left':0.5,'right':0.5,'angle':0.0}))