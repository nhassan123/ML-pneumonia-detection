from src.utils import *
from src.dataset import MedicalImagesDataset
from src.model import create_model, train
import cv2

NUM_CLASSES = 2

csv_train_file = "./data/stage_2_train_labels.csv"
csv_data_images_folder = "./data"

def visualize(label, image, bboxes):
    for box in bboxes:
        x, y, w, h = box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    message = "Pneumonia detected" if label == 1 else "Lung normal"
    cv2.putText(image, message, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)    
        
    cv2.imshow("Labelled Image", image)

    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows() # Destroy all OpenCV windows
    


if __name__ == '__main__':
    dataset = parse_csv(csv_train_file, csv_data_images_folder)
    (labels, images, bboxes) = create_tensor_from_dataset(dataset)
    trainDS = MedicalImagesDataset((images, labels, bboxes)) #can use a split to create a validation set

    model = create_model(num_classes=NUM_CLASSES)
    #model = model.to(DEVICE)
    
    train_loader = trainDS 
    train(train_loader, model)

    