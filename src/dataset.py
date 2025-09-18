from torch.utils.data import Dataset

class MedicalImagesDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms
    
    def __getitem__(self, index):
        image = self.tensors[0][index]
        target = {}
        target["label"] = self.tensors[1][index]
        target["boxes"] = self.tensors[2][index]
        
        print(image.shape)

        #transpose the image such that its channel dimmension becomes the leading one
        #image = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

        #transform if there are any transformations to apply
        if self.transforms:
            image = self.transforms(image)
        
        return (image, target)
    
    def __len__(self):
        return len(self.tensors[0])
