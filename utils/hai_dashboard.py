# Copyright 2020 Headlight AI Limited or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Transform txt file to json
def per_frame_analysis(data, threshold):
    data = slice_per(data, 7)   
    
    # Get image ID, eg: 0001.png
    def get_image_ID(frame):
        return frame[0]
    
    # Get timestamp of frame
    def get_timestamp(frame):
        return frame[1]
    
    # Get detections and return json holding relevant information
    def get_detections(frame, threshold):
        
        current_frame = {}
        
        ID = get_image_ID(frame)
        timestamp = get_timestamp(frame)
        
        current_frame['imageID'] = ID
        current_frame['timestamp'] = timestamp
        
        current_frame['probability'] = []
        current_frame['className'] = []
        current_frame['classID'] = []
        
        if len(frame) == 7:
            
            for detection in frame[2:]:
                
                detection_split = detection.split(',')

                probability = float(detection_split[0].split('=')[-1])
                
                if probability < threshold:
                    continue
                
                cls_name = detection_split[1].split(' ')[-1]
                cls_ID = detection_split[1].split(' ')[1].split('=')[-1]
                
                current_frame['probability'].append(probability)
                current_frame['className'].append(cls_name)
                current_frame['classID'].append(cls_ID)
                current_frame['timestamp'] = float(current_frame['timestamp'])
                
            return current_frame
                
        else:
            return None
                
    list_of_json = []
    for frame in data:
        
        frame_json = get_detections(frame, threshold)
        
        if frame_json == None:
            continue
        else:
            list_of_json.append(frame_json)
            
    return list_of_json


def slice_per(source, step):
    return [source[i:i+step] for i in range(0, len(source), step)]


def get_per_class_detections(data):
    

    per_class_detection = {}
    for detection in np.unique(data['detections']):
        if len(detection) !=0:
            per_class_detection[detection[0]] = []

            for idx, value in enumerate(data['detections']):
                if detection[0] in value:
                    per_class_detection[detection[0]].append(1)
                else:
                    per_class_detection[detection[0]].append(0)


    keys = [key for key in per_class_detection.keys()]
    per_class_detection['x'] = np.arange(len(per_class_detection[keys[0]]))
    return per_class_detection




def visualise(json_list):
    
    number_of_detection = []
    data = {'detections': [], 'time': [], 'number_of_detection': [], 'x': None}
    
    for entry in json_list:
        
        data['number_of_detection'].append(len(entry['probability']))
        data['detections'].append(entry['className'])
        data['time'].append(entry['timestamp'])
        
    data['time'] = np.array(data['time']).astype(float)
    sns.lineplot(x=np.arange(len(data['time'])), y = data['time'])
    plt.title('Inference time per frame (s)')
    plt.xlabel('Frame')
    plt.ylabel('Time (s)')
    plt.show()
    
        
        
        
    data['number_of_detection'] = np.array(data['number_of_detection']).astype(float)   
    per_class_detection = get_per_class_detections(data)
    
    for key in per_class_detection.keys():
        if key != 'x':
            
            sns.lineplot(x='x', y = key, data = per_class_detection, label = key)
            
            
    plt.title('Detections per frame')
    plt.xlabel('Frame')
    plt.ylabel('Detection present')
    plt.legend(loc='upper right')
    plt.show()
            
            
            
if __name__ == "__main__":
    
    with open('response.txt', 'r') as fp:
        data = fp.read().split('\n')

    list_of_json = per_frame_analysis(data, 0.8)

    data = visualise(list_of_json)

