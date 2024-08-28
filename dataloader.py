from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import torch
import numpy as np
import pandas as pd

class CUB(Dataset):
    def __init__(self, data_dir=Path('data/CUB_200_2011'), transform=None, train=True, majority_voting=False, concept_threshold=0):
        """
        Initialize the Caltech US Bird dataset.
        
        Args:
            data_dir (Path): Path to the CUB_200_2011 dataset.
            transform (callable): Transformations to apply to the images.
            train (bool): Whether to use the training set (True) or test set (False).
            majority_voting (bool): Whether to apply majority voting to concepts.
            concept_threshold (float): Threshold for filtering concepts.
        """
        super(CUB, self).__init__()

        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train = train

        # Load the dataset files
        self.images = np.loadtxt(data_dir / 'images.txt', dtype=str)
        self.split = np.loadtxt(data_dir / 'train_test_split.txt', dtype=int)[:,1]
        self.concepts = self.make_concept_list()
        self.original_concepts = self.concepts.copy()
        self.class_labels = np.loadtxt(data_dir / 'image_class_labels.txt', dtype=int)

        # Determine the number of classes dynamically
        self.num_classes = len(np.unique(self.class_labels[:, 1]))

        # Apply preprocessing steps if specified
        if majority_voting:
            self.concepts = self.majority_voting()

        if concept_threshold > 0:
            self.concepts, self.concepts_idx = self.filter_concepts(concept_threshold, majority_voting)
        else:
            self.concepts_idx = np.arange(self.concepts.shape[1])

        # Split the dataset into train and test sets
        if self.train:
            self.images = self.images[self.split == 1]
            self.class_labels = self.class_labels[self.split == 1]
            self.concepts = torch.tensor(self.concepts[self.split == 1], dtype=torch.float32)
        else:
            self.images = self.images[self.split == 0]
            self.class_labels = self.class_labels[self.split == 0]
            self.concepts = torch.tensor(self.concepts[self.split == 0], dtype=torch.float32)

    def make_concept_list(self):
        """
        Create a numpy array of the concepts for each image.
        
        Returns:
            numpy.ndarray: Array of concepts for all images.
        """
        concept_file = self.data_dir / 'attributes' / 'image_attribute_labels.txt'
        concepts = np.loadtxt(concept_file, dtype=int,usecols=(0, 1, 2))
        num_images = len(self.images)
        num_concepts = len(concepts) // num_images
        return concepts[:, 2].reshape(num_images, num_concepts)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch the image, concepts, and label for a given index.
        
        Args:
            idx (int): Index of the data point to fetch.
        
        Returns:
            tuple: (image, concepts, label)
        """
        image_path = self.data_dir / 'images' / self.images[idx][1]
        image = read_image(str(image_path))

        # If image is grayscale, convert to 3 channels 
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        concepts = self.concepts[idx]

        # Create one-hot encoding of the class label
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[self.class_labels[idx][1] - 1] = 1.0

        return image, concepts, label
    
    def majority_voting(self):
        """
        Apply majority voting to concepts based on class labels.
        This assigns the most common concept values for each class to all instances of that class.
        
        Returns:
            numpy.ndarray: Updated concepts after majority voting.
        """
        df = pd.DataFrame(self.concepts)
        df['class'] = self.class_labels[:,1]
        majority = df.groupby('class').mean().round().values
        return majority[self.class_labels[:,1] - 1]
    
    def filter_concepts(self, threshold, majority_voting):
        """
        Filter concepts based on their prevalence in the dataset.
        
        This function assumes that majority voting has already been applied if majority_voting is True.
        
        Args:
            threshold (float): Consept with less than trheshold prevalence will be removed.
                If majority_voting is True, concepts are grouped by class and the threshold is applied to the class prevalence in each class.
                If False, the threshold is applied to the overall prevalence of each concept in each image.
            majority_voting (bool): Indicates whether majority voting has been applied.
        
        Returns:
            numpy.ndarray: Filtered concepts, where each column represents a concept that meets the threshold criteria
            numpy.ndarray: Indices of the kept concepts
        """
        if majority_voting:
            df = pd.DataFrame(self.concepts)
            df['class'] = self.class_labels[:,1]
            prevalence = df.groupby('class').mean().sum(axis=0).values
            keep_concepts = prevalence >= threshold

        else:
            prevalence = self.concepts.sum(axis=0)
            keep_concepts = prevalence >= threshold


        return self.concepts[:, keep_concepts],np.arange(self.concepts.shape[1])[keep_concepts]
        
def get_dataset(dataset_name: str, data_path: str, train: bool = True, majority_voting: bool = False):
    if dataset_name.lower() == "cub":
        return CUB(data_dir=Path(data_path), train=train, majority_voting=majority_voting)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    


