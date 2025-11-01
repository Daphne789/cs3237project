import os
import numpy as np
curr_frames = list(os.scandir("captured_frames"))

sorted_curr_frames = []

for i in range(len(curr_frames)):
    if curr_frames[i].name == '.DS_Store':
        continue
    sorted_curr_frames.append(int(curr_frames[i].name.split('_')[1].split('.')[0]))

sorted_curr_frames.sort()

for i in range(len(sorted_curr_frames)):
    old_file_name = f"captured_frames/frame_{sorted_curr_frames[i]}.jpg"
    new_file_name = f"captured_frames/frame_{i}.jpg"
    # old_file_name = f"captured_frames_2/frame_{sorted_curr_frames[i]}.jpg"
    # new_file_name = f"captured_frames_2/frame_{int(sorted_curr_frames[i]) + 100000000}.jpg"
    os.rename(old_file_name, new_file_name)
    #print(old_file_name, new_file_name)

# for frame in curr_frames:
#     old_name = frame.name
#     new_name = os.path.join("captured_frames", old_name.split('_', 1)[1])
#     os.rename(os.path.join("captured_frames", old_name), new_name)