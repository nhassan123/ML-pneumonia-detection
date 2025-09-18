import pandas as pd
import numpy as np
import pydicom 

RESIZE_TO = 512

def parse_csv(csv_file, data_folder):
    dataset = pd.read_csv(csv_file)
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}

    for n, row in dataset.iterrows():
        patient_id = row['patientId']
        if patient_id not in parsed:
            parsed[patient_id] = {
                'dicom': '%s/stage_2_train_images/%s.dcm' % (data_folder, patient_id),
                'label': row['Target'],
                'boxes': []
            }
        
        if parsed[patient_id]['label'] == 1:
            parsed[patient_id]['boxes'].append(extract_box(row))
    
    return parsed

def create_tensor_from_dataset(parsed_dataset):
    images = []
    labels = []
    bboxes = []

    count = 0
    for patient in parsed_dataset:
        images.append(getImage(parsed_dataset[patient]['dicom']))
        labels.append(parsed_dataset[patient]['label'])
        #print(parsed_dataset[patient]['boxes'])
        bboxes.append(np.array(parsed_dataset[patient]['boxes'])) #bboxes may need to be adjusted based on image resizing
    
        if count == 10:
            break
        count += 1

    images = np.array(images)
    labels = np.array(labels)
    #bboxes = np.array(bboxes)

    return (labels, images, bboxes)

def getImage(dicom_file):
    #add line to resize image to 300x300 to save on CPU and RAM
    d = pydicom.dcmread(dicom_file)
    im = d.pixel_array
    # spacing_h = im.shape[0]//RESIZE_TO
    # spacing_w = im.shape[1]//RESIZE_TO
    # im = im[::spacing_h, ::spacing_w] 
    #convert single-channel gray scale image to 3 channels
    im = np.stack([im]*3, axis = 2) 
    return im