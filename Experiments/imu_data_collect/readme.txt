This project helps in tracking IMU data for various motions with the output being a .csv file. See the example output .csv file.

=== Usage ===

1. Build the hardware (see section below)
2. Upload .ino code to ESP32.
3. Close the IDE so that the port is not in use.
4. Start the .py file in the consol, providing the motion being tracked. Eg. "python imu_data_collect.py left"
5. Hold button to start tracking a motion. Release button to stop the tracking of the motion. Each button press corresponds to one motion tracked with many gyro and accel data of same action_id.
6. ctrl + c to stop the tracking of motions. The .csv file should be accessible in the same directory as the .py file. 

=== Hardware ===

LED and buzzer are optional, used for audio and visual feedback to know when tracking start or stop.

Led connect to P13 + GND

Buzzer connect to P26 + GND (The VCC is not necessary)

Button connect to P14 + GND

IMU
- VCC ---> 3.3v or 5v
- GND
- SCL ---> P22
- SDA ---> P21
- XDA (Not used)
- XCL (Not used)
- ADD (Not used)
- INT (Not used)
