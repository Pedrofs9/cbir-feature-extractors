# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from random import randint
import pickle

# PyTorch Imports
import torch
from torch.utils.data import Dataset

# Sklearn and Skmultilearn Imports
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split



# Class: ISIC2018Dataset
class ISIC2018Dataset(Dataset):

    def __init__(self, images_dir, metadata_dir, task, subset, transform=None, tr_perc=1.0, return_id=False):


        assert task in (1, 2, 3), "Please select a valid task (1, 2, 3)."
        assert subset in ('train', 'validation', 'test'), "Please select a valid subset (train, validation, test)"
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."


        # Assign variables
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.task1_dir = 'task1'
        self.task2_dir = 'task2'
        self.task3_dir = 'task3'
        self.tr_perc = tr_perc
        self.return_id = return_id

        # Task 1 Subset Directories
        self.task1_train_input_dir = 'ISIC2018_Task1-2_Training_Input'
        self.task1_train_groundtruth_dir = 'ISIC2018_Task1_Training_GroundTruth'
        self.task1_validation_input_dir = 'ISIC2018_Task1-2_Validation_Input'
        self.task1_validation_groundtruth_dir = 'ISIC2018_Task1_Validation_GroundTruth'
        self.task1_test_input_dir = 'ISIC2018_Task1-2_Test_Input'
        self.task1_test_groundtruth_dir = 'ISIC2018_Task1_Test_GroundTruth'
        self.task1_train_input_imgs = None
        self.task1_train_groundtruth_masks = None
        self.task1_validation_input_imgs = None
        self.task1_validation_groundtruth_masks = None
        self.task1_test_input_imgs = None
        self.task1_test_groundtruth_masks = None

        # Task 2 Subset Directories
        self.task2_train_input_dir = 'ISIC2018_Task1-2_Training_Input'
        self.task2_train_groundtruth_dir = 'ISIC2018_Task2_Training_GroundTruth_v3'
        self.task2_validation_input_dir = 'ISIC2018_Task1-2_Validation_Input'
        self.task2_validation_groundtruth_dir = 'ISIC2018_Task2_Validation_GroundTruth'
        self.task2_test_input_dir = 'ISIC2018_Task1-2_Test_Input'
        self.task2_test_groundtruth_dir = 'ISIC2018_Task2_Test_GroundTruth'
        self.task2_train_input_imgs = None
        self.task2_train_groundtruth_masks = None
        self.task2_validation_input_imgs = None
        self.task2_validation_groundtruth_masks = None
        self.task2_test_input_imgs = None
        self.task2_test_groundtruth_masks = None

        # Task 3 Subset Directories
        self.task3_train_input_dir = 'ISIC2018_Task3_Training_Input'
        self.taskt3_train_groundtruth_dir = 'ISIC2018_Task3_Training_GroundTruth'
        self.task3_validation_input_dir = 'ISIC2018_Task3_Validation_Input'
        self.task3_validation_groundtruth_dir = 'ISIC2018_Task3_Validation_GroundTruth'
        self.task3_test_input_dir = 'ISIC2018_Task3_Test_Input'
        self.task3_test_groundtruth_dir = 'ISIC2018_Task3_Test_GroundTruth'
        self.task3_train_input_imgs = None
        self.task3_train_labels = None
        self.task3_validation_input_imgs = None
        self.task3_validation_labels = None
        self.task3_test_input_imgs = None
        self.task3_test_labels = None
        self.task3_train_lesion_groupings = None


        # Load the data according to the data
        if task == 1:
            self.load_task1_data()
            images = list()
            masks = list()
            
            if subset == 'train':
                for image_fname in self.task1_train_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_train_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)

            elif subset == 'validation':
                for image_fname in self.task1_validation_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_validation_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)

            else:
                for image_fname in self.task1_test_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_test_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)
            
            self.images = images
            self.masks = masks
            self.labels = None
        
        
        elif task == 2:
            self.load_task2_data()
            images = list()
            masks_g = list()
            masks_m = list()
            masks_n = list()
            masks_p = list()
            masks_s = list()
            
            if subset == 'train':
                for image_fname in self.task2_train_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)

            elif subset == 'validation':
                for image_fname in self.task2_validation_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)

            else:
                for image_fname in self.task2_test_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)
            
            self.images = images
            self.masks = [masks_g, masks_m, masks_n, masks_p, masks_s]
            self.labels = None


        else:
            self.load_task3_data()
            images = list()
            labels = list()

            if subset == 'train':
                for _, row in self.task3_train_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_train_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)

            elif subset == 'validation':
                for _, row in self.task3_validation_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_validation_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)

            else:
                for _, row in self.task3_test_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_test_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)
            
            self.images = images
            self.labels = labels
            self.masks = None
        


        # Assign reamining variables
        self.task = task
        self.subset = subset
        self.transform = transform

        return
    

    # Method: Load Task 1 Data
    def load_task1_data(self):

        # print("Task 1")

        # Train
        task1_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_train_input_dir)) if not i.startswith('.')]
        task1_train_input_imgs = [i for i in task1_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task1_train_input_imgs)}")

        task1_train_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_train_groundtruth_dir)) if not i.startswith('.')]
        task1_train_groundtruth_masks = [i for i in task1_train_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Masks: {len(task1_train_groundtruth_masks)}")


        # Validation
        task1_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_input_dir)) if not i.startswith('.')]
        task1_validation_input_imgs = [i for i in task1_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task1_validation_input_imgs)}")

        task1_validation_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_groundtruth_dir)) if not i.startswith('.')]
        task1_validation_groundtruth_masks = [i for i in task1_validation_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Masks: {len(task1_validation_groundtruth_masks)}")


        # Test
        task1_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_test_input_dir)) if not i.startswith('.')]
        task1_test_input_imgs = [i for i in task1_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task1_test_input_imgs)}")

        task1_test_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_test_groundtruth_dir)) if not i.startswith('.')]
        task1_test_groundtruth_masks = [i for i in task1_test_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Masks: {len(task1_test_groundtruth_masks)}")


        # Assign variables
        self.task1_train_input_imgs = task1_train_input_imgs
        self.task1_train_groundtruth_masks = task1_train_groundtruth_masks
        self.task1_validation_input_imgs = task1_validation_input_imgs
        self.task1_validation_groundtruth_masks = task1_validation_groundtruth_masks
        self.task1_test_input_imgs = task1_test_input_imgs
        self.task1_test_groundtruth_masks = task1_test_groundtruth_masks

        return task1_train_input_imgs, task1_train_groundtruth_masks, task1_validation_input_imgs, task1_validation_groundtruth_masks, task1_test_input_imgs, task1_test_groundtruth_masks
    

    # Method: Load Task 2 Data
    def load_task2_data(self):

        # print("Task 2")

        # Train
        task2_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_train_input_dir)) if not i.startswith('.')]
        task2_train_input_imgs = [i for i in task2_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task2_train_input_imgs)}")

        task2_train_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir)) if not i.startswith('.')]
        task2_train_groundtruth_masks = [i for i in task2_train_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Masks: {len(task2_train_groundtruth_masks)}")


        # Validation
        task2_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_validation_input_dir)) if not i.startswith('.')]
        task2_validation_input_imgs = [i for i in task2_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task2_validation_input_imgs)}")

        task2_validation_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir)) if not i.startswith('.')]
        task2_validation_groundtruth_masks = [i for i in task2_validation_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Masks: {len(task2_validation_groundtruth_masks)}")


        # Test
        task2_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_test_input_dir)) if not i.startswith('.')]
        task2_test_input_imgs = [i for i in task2_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task2_test_input_imgs)}")

        task2_test_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir)) if not i.startswith('.')]
        task2_test_groundtruth_masks = [i for i in task2_test_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Masks: {len(task2_test_groundtruth_masks)}")


        # Assign variables
        self.task2_train_input_imgs = task2_train_input_imgs
        self.task2_train_groundtruth_masks = task2_train_groundtruth_masks
        self.task2_validation_input_imgs = task2_validation_input_imgs
        self.task2_validation_groundtruth_masks = task2_validation_groundtruth_masks
        self.task2_test_input_imgs = task2_test_input_imgs
        self.task2_test_groundtruth_masks = task2_test_groundtruth_masks

        return task2_train_input_imgs, task2_train_groundtruth_masks, task2_validation_input_imgs, task2_validation_groundtruth_masks, task2_test_input_imgs, task2_test_groundtruth_masks
    

    # Method: Load Task 3 Data
    def load_task3_data(self):

        # print("Task 3")

        # Train
        task3_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_train_input_dir)) if not i.startswith('.')]
        task3_train_input_imgs = [i for i in task3_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task3_train_input_imgs)}")

        task3_train_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.taskt3_train_groundtruth_dir, f"{self.taskt3_train_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        if self.tr_perc < 1.0 and self.tr_perc > 0.0:
            labels = task3_train_labels.apply(lambda x: np.argmax(np.array([x.MEL, x.NV, x.BCC, x.AKIEC, x.BKL, x.DF, x.VASC])), axis=1)
            task3_train_input_imgs, _ = train_test_split(task3_train_input_imgs, train_size=self.tr_perc, random_state=42, stratify=labels.values)
            task3_train_labels['image_jpg'] = task3_train_labels.apply(lambda x: x['image'] + '.jpg', axis=1)
            task3_train_labels = task3_train_labels[task3_train_labels['image_jpg'].isin(task3_train_input_imgs)]
        # print(f"Train Labels shape: {task3_train_labels.shape}")
        # print(task3_train_labels.head())


        # Validation
        task3_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_validation_input_dir)) if not i.startswith('.')]
        task3_validation_input_imgs = [i for i in task3_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task3_validation_input_imgs)}")

        task3_validation_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.task3_validation_groundtruth_dir, f"{self.task3_validation_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        # print(f"Validation Labels shape: {task3_validation_labels.shape}")
        # print(task3_validation_labels.head())


        # Test
        task3_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_test_input_dir)) if not i.startswith('.')]
        task3_test_input_imgs = [i for i in task3_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task3_test_input_imgs)}")

        task3_test_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.task3_test_groundtruth_dir, f"{self.task3_test_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        # print(f"Test Labels shape: {task3_test_labels.shape}")
        # print(task3_test_labels.head())


        # Lesion Groupings
        task3_train_lesion_groupings = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, 'ISIC2018_Task3_Training_LesionGroupings.csv'),
            delimiter=',',
            header=0
        )
        # print(f"Task 3 Train Lesion Groupings shape: {task3_train_lesion_groupings.shape}")
        # print(task3_train_lesion_groupings.head())


        # Assign variables
        self.task3_train_input_imgs = task3_train_input_imgs
        self.task3_train_labels = task3_train_labels
        self.task3_validation_input_imgs = task3_validation_input_imgs
        self.task3_validation_labels = task3_validation_labels
        self.task3_test_input_imgs = task3_test_input_imgs
        self.task3_test_labels = task3_test_labels
        self.task3_train_lesion_groupings = task3_train_lesion_groupings


        return task3_train_input_imgs, task3_train_labels, task3_validation_input_imgs, task3_validation_labels, task3_test_input_imgs, task3_test_labels, task3_train_lesion_groupings


    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Task 1
        if self.task == 1:
            image_fname = self.images[idx]
            mask_fname = self.masks[idx]
            
            if self.subset == 'train':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_train_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_train_groundtruth_dir, mask_fname)).convert('L')
            elif self.subset == 'validation':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_groundtruth_dir, mask_fname)).convert('L')
            else:
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_test_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_test_groundtruth_dir, mask_fname)).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            if self.return_id:
                return image, mask, image_fname.replace(".jpg", "")
            return image, mask



        # Task 2
        elif self.task == 2:
            image_fname = self.images[idx]
            mask_g_fname = self.masks[0][idx]
            mask_m_fname = self.masks[1][idx]
            mask_n_fname = self.masks[2][idx]
            mask_p_fname = self.masks[3][idx]
            mask_s_fname = self.masks[4][idx]

            if self.subset == 'train':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_train_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_s_fname)).convert('L')

            elif self.subset == 'validation':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_validation_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_s_fname)).convert('L')

            else:
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_test_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_s_fname)).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask_g = self.transform(mask_g)
                mask_m = self.transform(mask_m)
                mask_n = self.transform(mask_n)
                mask_p = self.transform(mask_p)
                mask_s = self.transform(mask_s)
            
            if self.return_id:
                return image, mask_g, mask_m, mask_n, mask_p, mask_s, image_fname.replace(".jpg", "")
            return image, mask_g, mask_m, mask_n, mask_p, mask_s



        # Task 3
        else:
            image_fname = self.images[idx]
            label = np.array(self.labels[idx], dtype=np.int64).argmax(-1)

            if self.subset == 'train':
                image = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_train_input_dir, image_fname)).convert('RGB')
            elif self.subset == 'validation':
                image = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_validation_input_dir, image_fname)).convert('RGB')
            else:
                image = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_test_input_dir, image_fname)).convert('RGB')


            if self.transform:
                image = self.transform(image)
            
            # Convert label to tensor
            label = torch.as_tensor(label, dtype=torch.int64)
            
            if self.return_id:
                return image, label, image_fname.replace(".jpg", "")
            return image, label



# Class:ISIC2018CINetDataset
class ISIC2018CINetDataset(Dataset):

    def __init__(self, images_dir, metadata_dir, task, subset, transform=None, tr_perc=1.0, return_id=False):


        assert task in (1, 2, 3), "Please select a valid task (1, 2, 3)."
        assert subset in ('train', 'validation', 'test'), "Please select a valid subset (train, validation, test)"
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."


        # Assign variables
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.task1_dir = 'task1'
        self.task2_dir = 'task2'
        self.task3_dir = 'task3'
        self.tr_perc = tr_perc
        self.return_id = return_id

        # Task 1 Subset Directories
        self.task1_train_input_dir = 'ISIC2018_Task1-2_Training_Input'
        self.task1_train_groundtruth_dir = 'ISIC2018_Task1_Training_GroundTruth'
        self.task1_validation_input_dir = 'ISIC2018_Task1-2_Validation_Input'
        self.task1_validation_groundtruth_dir = 'ISIC2018_Task1_Validation_GroundTruth'
        self.task1_test_input_dir = 'ISIC2018_Task1-2_Test_Input'
        self.task1_test_groundtruth_dir = 'ISIC2018_Task1_Test_GroundTruth'
        self.task1_train_input_imgs = None
        self.task1_train_groundtruth_masks = None
        self.task1_validation_input_imgs = None
        self.task1_validation_groundtruth_masks = None
        self.task1_test_input_imgs = None
        self.task1_test_groundtruth_masks = None

        # Task 2 Subset Directories
        self.task2_train_input_dir = 'ISIC2018_Task1-2_Training_Input'
        self.task2_train_groundtruth_dir = 'ISIC2018_Task2_Training_GroundTruth_v3'
        self.task2_validation_input_dir = 'ISIC2018_Task1-2_Validation_Input'
        self.task2_validation_groundtruth_dir = 'ISIC2018_Task2_Validation_GroundTruth'
        self.task2_test_input_dir = 'ISIC2018_Task1-2_Test_Input'
        self.task2_test_groundtruth_dir = 'ISIC2018_Task2_Test_GroundTruth'
        self.task2_train_input_imgs = None
        self.task2_train_groundtruth_masks = None
        self.task2_validation_input_imgs = None
        self.task2_validation_groundtruth_masks = None
        self.task2_test_input_imgs = None
        self.task2_test_groundtruth_masks = None

        # Task 3 Subset Directories
        self.task3_train_input_dir = 'ISIC2018_Task3_Training_Input'
        self.taskt3_train_groundtruth_dir = 'ISIC2018_Task3_Training_GroundTruth'
        self.task3_validation_input_dir = 'ISIC2018_Task3_Validation_Input'
        self.task3_validation_groundtruth_dir = 'ISIC2018_Task3_Validation_GroundTruth'
        self.task3_test_input_dir = 'ISIC2018_Task3_Test_Input'
        self.task3_test_groundtruth_dir = 'ISIC2018_Task3_Test_GroundTruth'
        self.task3_train_input_imgs = None
        self.task3_train_labels = None
        self.task3_validation_input_imgs = None
        self.task3_validation_labels = None
        self.task3_test_input_imgs = None
        self.task3_test_labels = None
        self.task3_train_lesion_groupings = None


        # Load the data according to the data
        if task == 1:
            self.load_task1_data()
            images = list()
            masks = list()
            
            if subset == 'train':
                for image_fname in self.task1_train_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_train_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)

            elif subset == 'validation':
                for image_fname in self.task1_validation_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_validation_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)

            else:
                for image_fname in self.task1_test_input_imgs:
                    mask_fname = image_fname.split('.')[0] + '_segmentation.png'
                    assert mask_fname in self.task1_test_groundtruth_masks, f"There is a missing mask: {mask_fname}"
                    images.append(image_fname)
                    masks.append(mask_fname)
            
            self.images = images
            self.masks = masks
            self.labels = None
        
        
        elif task == 2:
            self.load_task2_data()
            images = list()
            masks_g = list()
            masks_m = list()
            masks_n = list()
            masks_p = list()
            masks_s = list()
            
            if subset == 'train':
                for image_fname in self.task2_train_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_train_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)

            elif subset == 'validation':
                for image_fname in self.task2_validation_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_validation_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)

            else:
                for image_fname in self.task2_test_input_imgs:
                    mask_g_fname = image_fname.split('.')[0] + '_attribute' + '_globules.png'
                    mask_m_fname = image_fname.split('.')[0] + '_attribute' + '_milia_like_cyst.png'
                    mask_n_fname = image_fname.split('.')[0] + '_attribute' + '_negative_network.png'
                    mask_p_fname = image_fname.split('.')[0] + '_attribute' + '_pigment_network.png'
                    mask_s_fname = image_fname.split('.')[0] + '_attribute' + '_streaks.png'
                    assert mask_g_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_g_fname}"
                    assert mask_m_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_m_fname}"
                    assert mask_n_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_n_fname}"
                    assert mask_p_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_p_fname}"
                    assert mask_s_fname in self.task2_test_groundtruth_masks, f"There is a missing mask: {mask_s_fname}"
                    images.append(image_fname)
                    masks_g.append(mask_g_fname)
                    masks_m.append(mask_m_fname)
                    masks_n.append(mask_n_fname)
                    masks_p.append(mask_p_fname)
                    masks_s.append(mask_s_fname)
            
            self.images = images
            self.masks = [masks_g, masks_m, masks_n, masks_p, masks_s]
            self.labels = None


        else:
            self.load_task3_data()
            images = list()
            labels = list()

            if subset == 'train':
                for _, row in self.task3_train_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_train_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)

            elif subset == 'validation':
                for _, row in self.task3_validation_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_validation_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)

            else:
                for _, row in self.task3_test_labels.iterrows():
                    image_fname = row['image'] + '.jpg'
                    label = row[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
                    assert image_fname in self.task3_test_input_imgs, f"There is a missing image: {image_fname}"
                    images.append(image_fname)
                    labels.append(label)
            
            self.images = images
            self.labels = labels
            self.masks = None
        


        # Assign reamining variables
        self.task = task
        self.subset = subset
        self.transform = transform

        return
    

    # Method: Load Task 1 Data
    def load_task1_data(self):

        # print("Task 1")

        # Train
        task1_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_train_input_dir)) if not i.startswith('.')]
        task1_train_input_imgs = [i for i in task1_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task1_train_input_imgs)}")

        task1_train_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_train_groundtruth_dir)) if not i.startswith('.')]
        task1_train_groundtruth_masks = [i for i in task1_train_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Masks: {len(task1_train_groundtruth_masks)}")


        # Validation
        task1_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_input_dir)) if not i.startswith('.')]
        task1_validation_input_imgs = [i for i in task1_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task1_validation_input_imgs)}")

        task1_validation_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_groundtruth_dir)) if not i.startswith('.')]
        task1_validation_groundtruth_masks = [i for i in task1_validation_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Masks: {len(task1_validation_groundtruth_masks)}")


        # Test
        task1_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_test_input_dir)) if not i.startswith('.')]
        task1_test_input_imgs = [i for i in task1_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task1_test_input_imgs)}")

        task1_test_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task1_test_groundtruth_dir)) if not i.startswith('.')]
        task1_test_groundtruth_masks = [i for i in task1_test_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Masks: {len(task1_test_groundtruth_masks)}")


        # Assign variables
        self.task1_train_input_imgs = task1_train_input_imgs
        self.task1_train_groundtruth_masks = task1_train_groundtruth_masks
        self.task1_validation_input_imgs = task1_validation_input_imgs
        self.task1_validation_groundtruth_masks = task1_validation_groundtruth_masks
        self.task1_test_input_imgs = task1_test_input_imgs
        self.task1_test_groundtruth_masks = task1_test_groundtruth_masks

        return task1_train_input_imgs, task1_train_groundtruth_masks, task1_validation_input_imgs, task1_validation_groundtruth_masks, task1_test_input_imgs, task1_test_groundtruth_masks
    

    # Method: Load Task 2 Data
    def load_task2_data(self):

        # print("Task 2")

        # Train
        task2_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_train_input_dir)) if not i.startswith('.')]
        task2_train_input_imgs = [i for i in task2_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task2_train_input_imgs)}")

        task2_train_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir)) if not i.startswith('.')]
        task2_train_groundtruth_masks = [i for i in task2_train_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Masks: {len(task2_train_groundtruth_masks)}")


        # Validation
        task2_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_validation_input_dir)) if not i.startswith('.')]
        task2_validation_input_imgs = [i for i in task2_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task2_validation_input_imgs)}")

        task2_validation_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir)) if not i.startswith('.')]
        task2_validation_groundtruth_masks = [i for i in task2_validation_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Masks: {len(task2_validation_groundtruth_masks)}")


        # Test
        task2_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task1_dir, self.task2_test_input_dir)) if not i.startswith('.')]
        task2_test_input_imgs = [i for i in task2_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task2_test_input_imgs)}")

        task2_test_groundtruth_masks = [i for i in os.listdir(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir)) if not i.startswith('.')]
        task2_test_groundtruth_masks = [i for i in task2_test_groundtruth_masks if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Masks: {len(task2_test_groundtruth_masks)}")


        # Assign variables
        self.task2_train_input_imgs = task2_train_input_imgs
        self.task2_train_groundtruth_masks = task2_train_groundtruth_masks
        self.task2_validation_input_imgs = task2_validation_input_imgs
        self.task2_validation_groundtruth_masks = task2_validation_groundtruth_masks
        self.task2_test_input_imgs = task2_test_input_imgs
        self.task2_test_groundtruth_masks = task2_test_groundtruth_masks

        return task2_train_input_imgs, task2_train_groundtruth_masks, task2_validation_input_imgs, task2_validation_groundtruth_masks, task2_test_input_imgs, task2_test_groundtruth_masks
    

    # Method: Load Task 3 Data
    def load_task3_data(self):

        # print("Task 3")

        # Train
        task3_train_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_train_input_dir)) if not i.startswith('.')]
        task3_train_input_imgs = [i for i in task3_train_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Train Images: {len(task3_train_input_imgs)}")

        task3_train_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.taskt3_train_groundtruth_dir, f"{self.taskt3_train_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        if self.tr_perc < 1.0 and self.tr_perc > 0.0:
            labels = task3_train_labels.apply(lambda x: np.argmax(np.array([x.MEL, x.NV, x.BCC, x.AKIEC, x.BKL, x.DF, x.VASC])), axis=1)
            task3_train_input_imgs, _ = train_test_split(task3_train_input_imgs, train_size=self.tr_perc, random_state=42, stratify=labels.values)
            task3_train_labels['image_jpg'] = task3_train_labels.apply(lambda x: x['image'] + '.jpg', axis=1)
            task3_train_labels = task3_train_labels[task3_train_labels['image_jpg'].isin(task3_train_input_imgs)]
        # print(f"Train Labels shape: {task3_train_labels.shape}")
        # print(task3_train_labels.head())


        # Validation
        task3_validation_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_validation_input_dir)) if not i.startswith('.')]
        task3_validation_input_imgs = [i for i in task3_validation_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Validation Images: {len(task3_validation_input_imgs)}")

        task3_validation_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.task3_validation_groundtruth_dir, f"{self.task3_validation_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        # print(f"Validation Labels shape: {task3_validation_labels.shape}")
        # print(task3_validation_labels.head())


        # Test
        task3_test_input_imgs = [i for i in os.listdir(os.path.join(self.images_dir, self.task3_dir, self.task3_test_input_dir)) if not i.startswith('.')]
        task3_test_input_imgs = [i for i in task3_test_input_imgs if i.split('.')[1].lower()!='txt']
        # print(f"Nr. Test Images: {len(task3_test_input_imgs)}")

        task3_test_labels = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, self.task3_test_groundtruth_dir, f"{self.task3_test_groundtruth_dir}.csv"),
            delimiter=',',
            header=0
        )
        # print(f"Test Labels shape: {task3_test_labels.shape}")
        # print(task3_test_labels.head())


        # Lesion Groupings
        task3_train_lesion_groupings = pd.read_csv(
            filepath_or_buffer=os.path.join(self.metadata_dir, 'ISIC2018_Task3_Training_LesionGroupings.csv'),
            delimiter=',',
            header=0
        )
        # print(f"Task 3 Train Lesion Groupings shape: {task3_train_lesion_groupings.shape}")
        # print(task3_train_lesion_groupings.head())


        # Assign variables
        self.task3_train_input_imgs = task3_train_input_imgs
        self.task3_train_labels = task3_train_labels
        self.task3_validation_input_imgs = task3_validation_input_imgs
        self.task3_validation_labels = task3_validation_labels
        self.task3_test_input_imgs = task3_test_input_imgs
        self.task3_test_labels = task3_test_labels
        self.task3_train_lesion_groupings = task3_train_lesion_groupings


        return task3_train_input_imgs, task3_train_labels, task3_validation_input_imgs, task3_validation_labels, task3_test_input_imgs, task3_test_labels, task3_train_lesion_groupings


    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Task 1
        if self.task == 1:
            image_fname = self.images[idx]
            mask_fname = self.masks[idx]
            
            if self.subset == 'train':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_train_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_train_groundtruth_dir, mask_fname)).convert('L')
            elif self.subset == 'validation':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_validation_groundtruth_dir, mask_fname)).convert('L')
            else:
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_test_input_dir, image_fname)).convert('RGB')
                mask = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task1_test_groundtruth_dir, mask_fname)).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            if self.return_id:
                return image, mask, image_fname.replace(".jpg", "")
            
            return image, mask



        # Task 2
        elif self.task == 2:
            image_fname = self.images[idx]
            mask_g_fname = self.masks[0][idx]
            mask_m_fname = self.masks[1][idx]
            mask_n_fname = self.masks[2][idx]
            mask_p_fname = self.masks[3][idx]
            mask_s_fname = self.masks[4][idx]

            if self.subset == 'train':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_train_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_train_groundtruth_dir, mask_s_fname)).convert('L')

            elif self.subset == 'validation':
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_validation_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_validation_groundtruth_dir, mask_s_fname)).convert('L')

            else:
                image = Image.open(os.path.join(self.images_dir, self.task1_dir, self.task2_test_input_dir, image_fname)).convert('RGB')
                mask_g = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_g_fname)).convert('L')
                mask_m = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_m_fname)).convert('L')
                mask_n = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_n_fname)).convert('L')
                mask_p = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_p_fname)).convert('L')
                mask_s = Image.open(os.path.join(self.images_dir, self.task2_dir, self.task2_test_groundtruth_dir, mask_s_fname)).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask_g = self.transform(mask_g)
                mask_m = self.transform(mask_m)
                mask_n = self.transform(mask_n)
                mask_p = self.transform(mask_p)
                mask_s = self.transform(mask_s)
            
            if self.return_id:
                return image, mask_g, mask_m, mask_n, mask_p, mask_s, image_fname.replace(".jpg", "")
            
            return image, mask_g, mask_m, mask_n, mask_p, mask_s



        # Task 3
        else:
            
            # Get the length of the images
            len_ = len(self.images)

            # Image 0
            image0_fname = self.images[idx]
            classlabel0 = np.array(self.labels[idx], dtype=np.int64).argmax(-1)

            # Image 1
            if idx != len_ - 1:
                a = randint(1, len_ - idx - 1)
                image1_fname = self.images[idx+a]
                classlabel1 = np.array(self.labels[idx+a], dtype=np.int64).argmax(-1)
            else:
                a = randint(1, len_ - 1)
                image1_fname = self.images[idx-a]
                classlabel1 = np.array(self.labels[idx-a], dtype=np.int64).argmax(-1)


            if self.subset == 'train':
                img0 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_train_input_dir, image0_fname)).convert('RGB')
                img1 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_train_input_dir, image1_fname)).convert('RGB')
            elif self.subset == 'validation':
                img0 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_validation_input_dir, image0_fname)).convert('RGB')
                img1 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_validation_input_dir, image1_fname)).convert('RGB')
            else:
                img0 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_test_input_dir, image0_fname)).convert('RGB')
                img1 = Image.open(os.path.join(self.images_dir, self.task3_dir, self.task3_test_input_dir, image1_fname)).convert('RGB')


            if self.transform:
                img0 = self.transform(img0)
                img1 = self.transform(img1)

            # Contrastive approach of CINet
            if classlabel0 == classlabel1:
                synlabel = 1
            else:
                synlabel = 0


            dislabel11 = 1
            dislabel12 = 0
            dislabel21 = 0
            dislabel22 = 1
            

            # Convert label to tensor
            classlabel0 = torch.as_tensor(classlabel0, dtype=torch.int64)
            classlabel1 = torch.as_tensor(classlabel1, dtype=torch.int64)
            
            if self.return_id:
                return img0, img1, classlabel0, classlabel1, synlabel, dislabel11, dislabel12, dislabel21, dislabel22, image0_fname.replace(".jpg", "")
            
            return img0, img1, classlabel0, classlabel1, synlabel, dislabel11, dislabel12, dislabel21, dislabel22



# Class: CXR8Dataset
class CXR8Dataset(Dataset):

    def __init__(self, images_dir, metadata_dir, subset, transform=None, tr_perc=1.0, return_id=False):

        assert subset in ('train', 'validation', 'test'), "Please select a valid subset (train, validation, test)"
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."

        # Assign variables
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.train_list = None
        self.val_list = None
        self.test_list = None
        self.bbox_list = None
        self.data_entry = None
        self.tr_perc = tr_perc
        self.return_id = return_id

        # Create a label index dictionary
        self.label_idx_dict = {
            'Atelectasis': 1,
            'Cardiomegaly': 2,
            'Effusion': 3,
            'Infiltration': 4,
            'Mass': 5,
            'Nodule': 6,
            'Pneumonia': 7,
            'Pneumothorax': 8,
            'Consolidation': 9,
            'Edema': 10,
            'Emphysema': 11,
            'Fibrosis': 12,
            'Pleural_Thickening': 13,
            'Hernia': 14,
            'No Finding': 15
        }

        # Load data entry
        self.load_data_entry()

        # Load data splits
        self.load_data_splits()

        # Load bbox list
        self.load_bbox_list()

        if subset == 'train':
            self.images = self.train_list.copy()
        elif subset == 'validation':
            self.images = self.val_list.copy()
        else:
            self.images = self.test_list.copy()

        self.labels = self.get_labels(self.images)
        self.transform = transform

        return


    # Method: Open data splits
    def load_data_splits(self):
        assert self.data_entry is not None

        # Check if train-val split exists
        if not(os.path.exists(os.path.join(self.metadata_dir, 'train_split.txt')) and os.path.exists(os.path.join(self.metadata_dir, 'val_split.txt'))):
            # Generate fixed train-val splits (to ensure they are always the same since the iterative_train_test_split doesn't have a random_state)
            train_val_list = pd.read_csv(os.path.join(self.metadata_dir, 'train_val_list.txt'), header=None)
            train_val_list = train_val_list[0].copy().values

            print("Creating train-val splits...")
            train_val_labels = self.get_labels(train_val_list)
            train_list, _, val_list, _ = iterative_train_test_split(np.array(train_val_list)[:, np.newaxis], train_val_labels, test_size=0.1)
            train_list = list(train_list.flatten())
            val_list = list(val_list.flatten())
            for fname, data in zip(['train_split.txt', 'val_split.txt'], [train_list, val_list]):
                with open(os.path.join(self.metadata_dir, fname), 'w') as f:
                    for line in data:
                        f.write(f"{line}\n")
            
        train_list = pd.read_csv(os.path.join(self.metadata_dir, 'train_split.txt'), header=None)
        train_list = train_list[0].copy().values

        if self.tr_perc < 1.0 and self.tr_perc > 0.0:
            # Check if train-val split exists
            if not os.path.exists(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt')):
                # Generate fixed train split (to ensure they are always the same since the iterative_train_test_split doesn't have a random_state)
                print(f"Creating train_{self.tr_perc} splits...")
                train_labels = self.get_labels(train_list)
                train_list, _, _, _ = iterative_train_test_split(np.array(train_list)[:, np.newaxis], train_labels, test_size=1.0-self.tr_perc)
                train_list = list(train_list.flatten())
                with open(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt'), 'w') as f:
                    for line in train_list:
                        f.write(f"{line}\n")
            train_list = pd.read_csv(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt'), header=None)
            train_list = train_list[0].copy().values

        val_list = pd.read_csv(os.path.join(self.metadata_dir, 'val_split.txt'), header=None)
        val_list = val_list[0].copy().values

        test_list = pd.read_csv(os.path.join(self.metadata_dir, 'test_list.txt'), header=None)
        test_list = test_list[0].copy().values

        # Assign to class variables
        self.train_list = train_list.copy()
        self.val_list = val_list.copy()
        self.test_list = test_list.copy()

        return train_list, val_list, test_list
    

    # Method: Load BBox list
    def load_bbox_list(self):
        
        # Load bbox list .CSV
        bbox_list = pd.read_csv(os.path.join(self.metadata_dir, 'BBox_List_2017.csv'), header=0)

        # Remove NaN columns
        bbox_list = bbox_list.copy().dropna(axis=1)

        # Assign to class variable
        self.bbox_list = bbox_list.copy()

        return bbox_list
    

    # Method: Load Data Entry
    def load_data_entry(self):

        # Load Data Entry from .CSV file
        data_entry = pd.read_csv(os.path.join(self.metadata_dir, "Data_Entry_2017_v2020.csv"), header=0)

        # Assign to class variable
        self.data_entry = data_entry.copy()

        return data_entry


    # Method: Get Labels
    def get_labels(self, imgs_list):
        assert self.data_entry is not None
        labels = []

        # Filter dataframe by imgs_list
        df = self.data_entry[self.data_entry['Image Index'].isin(imgs_list)]
        assert len(df) == len(imgs_list)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            
            finding_label = row['Finding Labels']
                
            # Create a label vector
            label = np.zeros(len(self.label_idx_dict))
            
            # Serialise finding_label
            finding_label = finding_label.split('|')

            for f_label in finding_label:
                label[self.label_idx_dict[f_label]-1] = 1
            
            labels.append(label)

        assert len(labels) == len(imgs_list)
        return np.array(labels)


    # Method: __len__
    def __len__(self):
        return len(self.images)
    

    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image fname
        image_fname = self.images[idx]
        
        # Get labels
        label = self.labels[idx]

        # Open image
        image = Image.open(os.path.join(self.images_dir, image_fname)).convert('RGB')

        # Apply image transfors
        if self.transform:
            image = self.transform(image)
        

        # Convert label to tensor
        label = torch.as_tensor(label).long()

        if self.return_id:
            return image, label, image_fname.replace(".png", "")
        
        return image, label



# Class: APTOSDataset
class APTOSDataset(Dataset):
    def __init__(self, images_dir, metadata_dir, subset, transform=None, tr_perc=1.0, return_id=False):

        assert subset in ('train', 'validation', 'test'), "Please provide a valid subset: train, validation or test."
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."

        data = pd.read_csv(os.path.join(metadata_dir, 'train.csv'))
        data_train, data_test = train_test_split(data, stratify=data['diagnosis'], test_size=0.2, random_state=42)
        data_train, data_val = train_test_split(data_train, stratify=data_train['diagnosis'], test_size=0.1, random_state=42)

        if tr_perc < 1.0 and tr_perc > 0.0:
            data_train, _ = train_test_split(data_train, stratify=data_train['diagnosis'], train_size=tr_perc, random_state=42)

        if subset == 'train':
            self.data = data_train.copy().reset_index()
        elif subset == 'validation':
            self.data = data_val.copy().reset_index()
        else:
            self.data = data_test.copy().reset_index()
        
        self.images_dir = images_dir
        self.transform = transform
        self.return_id = return_id

    # Method: __len__
    def __len__(self):

        return len(self.data)


     # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image fname
        image_fname = self.data.iloc[idx]['id_code']
        
        # Get labels
        label = self.data.iloc[idx]['diagnosis']

        # Open image
        image = Image.open(os.path.join(self.images_dir, f"{image_fname}.png")).convert('RGB')

        # Apply image transfors
        if self.transform:
            image = self.transform(image)
        

        # Convert label to tensor
        label = torch.as_tensor(label, dtype=torch.int64)

        if self.return_id:
            return image, label, image_fname

        return image, label
    
# Class: HerlevDataset
class HerlevDataset(Dataset):
    def __init__(self, pickle_file, subset, transform=None, tr_perc=1.0, return_id=False):

        assert subset in ('train', 'validation', 'test'), "Please provide a valid subset: train, validation or test."
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."

        data = pickle.load(open(pickle_file, 'rb'))
        imgs, labels, imgs_names = data["imgs"], data["labels"], data["imgs_names"]
        assert len(imgs) == len(labels)
        assert len(imgs) == len(imgs_names)
        ids = list(range(len(labels)))
        data_train, data_val = train_test_split(ids, stratify=labels, test_size=0.2, random_state=42)
        data_train, data_test = train_test_split(data_train, stratify=np.array(labels)[data_train], test_size=0.1, random_state=42)

        if tr_perc < 1.0 and tr_perc > 0.0:
            data_train, _ = train_test_split(data_train, stratify=np.array(labels)[data_train], train_size=tr_perc, random_state=42)

        if subset == 'train':
            self.imgs = np.array(imgs)[data_train]
            self.labels = np.array(labels)[data_train]
            self.imgs_names = np.array(imgs_names)[data_train]
        elif subset == 'validation':
            self.imgs = np.array(imgs)[data_val]
            self.labels = np.array(labels)[data_val]
            self.imgs_names = np.array(imgs_names)[data_val]
        else:
            self.imgs = np.array(imgs)[data_test]
            self.labels = np.array(labels)[data_test]
            self.imgs_names = np.array(imgs_names)[data_test]
        
        self.transform = transform
        self.return_id = return_id

    # Method: __len__
    def __len__(self):

        return len(self.imgs)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image fname
        image = self.imgs[idx]
        
        # Get labels
        label = self.labels[idx]

        # Apply image transfors
        if self.transform:
            image = self.transform(image)
        

        # Convert label to tensor
        label = torch.as_tensor(label, dtype=torch.int64)

        if self.return_id:
            return image, label, self.imgs_names[idx]

        return image, label



# Class: VinDrCXRDataset
class VinDrCXRDataset(Dataset):
    def __init__(self, images_dir="images-resized", metadata_dir="annotations-resized", subset="train", transform=None, tr_perc=1.0, return_id=False, use_global_labels=False):

        assert subset in ('train', 'validation', 'test'), "Please select a valid subset (train, validation, test)"
        assert tr_perc <= 1.0 and tr_perc >= 0.01, "The train percentage must be in between [0.01, 1.0]."

        # Assign variables
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.train_list = None
        self.val_list = None
        self.test_list = None
        self.tr_perc = tr_perc
        self.return_id = return_id
        self.use_global_labels = use_global_labels
        self.subset = subset
        self.return_bbox = False

        # Create a dictionary for labels
        self.all_labels_list = [
            "Aortic enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Clavicle fracture",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Enlarged PA",
            "ILD",
            "Infiltration",
            "Lung Opacity",
            "Lung cavity",
            "Lung cyst",
            "Mediastinal shift",
            "Nodule/Mass",
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis",
            "Rib fracture",
            "Other lesion",
            "COPD",
            "Lung tumor",
            "Pneumonia",
            "Tuberculosis",
            "Other diseases",
            "No finding"
        ]
        
        # Local labels
        self.local_names_to_labels_dict, self.local_labels_to_names_dict = dict(), dict()

        # Global labels
        self.global_names_to_labels_dict, self.global_labels_to_names_dict = dict(), dict()

        # Populate dictionaries
        for label_idx, label_name in enumerate(self.all_labels_list):
            self.global_names_to_labels_dict[label_name] = label_idx
            self.global_labels_to_names_dict[label_idx] = label_name

            if label_name not in ("COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", "No finding"):
                self.local_names_to_labels_dict[label_name] = label_idx
                self.local_labels_to_names_dict[label_idx] = label_name

        # Load annotations
        self.train_dict, self.test_dict = self.load_annotations()

        # Load data splits
        train_list, val_list, _ = self.load_data_splits()

        # Get the proper data dictionary
        if self.subset == "train":
            self.data_dict = self.train_dict[self.train_dict["image_id"].isin(train_list)]
        elif self.subset == "validation":
            self.data_dict = self.train_dict[self.train_dict["image_id"].isin(val_list)]
        else:
            self.data_dict = self.test_dict

        # Get transforms
        self.transform = transform

        return
    

    # Method: Load annotations 
    def load_annotations(self):

        # Load CSVs
        train_annotations, test_annotations = None, None
        for subset in ["train", "test"]:
            csv_fname = f"annotations_{subset}.csv"
            csv_fpath = os.path.join(self.metadata_dir, csv_fname)
            if os.path.exists(csv_fpath):
                annotations_df = pd.read_csv(csv_fpath)
                if "rad_id" in annotations_df.columns:
                    annotations_df = annotations_df.drop(["rad_id"], axis=1)
                annotations_df["bbox"] = annotations_df.apply(lambda row: (row["x_min"], row["y_min"], row["x_max"], row["y_max"]), axis=1)
                annotations_df = annotations_df.drop(["x_min", "y_min", "x_max", "y_max"], axis=1)
                annotations_df["class_idx"] = annotations_df.apply(lambda row: self.class_name_to_idx(row["class_name"]), axis=1)
                annotations_df = annotations_df.groupby("image_id").agg({"class_name": list, "class_idx": list, "bbox": list})
                annotations_df["image_label_vec"] = annotations_df.apply(lambda row: self.get_label_vector(row["class_idx"]), axis=1)
                annotations_df = annotations_df.reset_index()
                if subset == "train":
                    train_annotations = annotations_df.copy()
                else:
                    test_annotations = annotations_df.copy()

        return train_annotations, test_annotations

    # Method: Class name to idx
    def class_name_to_idx(self, class_name):
        if not self.use_global_labels:
            if class_name in self.local_names_to_labels_dict.keys() or class_name == "No finding":
                return -1 if class_name == "No finding" else int(self.local_names_to_labels_dict[class_name])
        else:
            return self.global_names_to_labels_dict[class_name]

    # Method: Get label vector
    def get_label_vector(self, labels):
        # Build label vectors
        res = np.zeros(len(self.global_names_to_labels_dict)) if self.use_global_labels else np.zeros(len(self.local_names_to_labels_dict))
        for label_idx in labels:
            if label_idx != -1:
                res[label_idx] = 1
        return res

    # Method: Open data splits
    def load_data_splits(self):

        # Check if train-val split exists
        if not(os.path.exists(os.path.join(self.metadata_dir, 'train_split.txt')) and os.path.exists(os.path.join(self.metadata_dir, 'val_split.txt'))):
            # Generate fixed train-val splits (to ensure they are always the same since the iterative_train_test_split doesn't have a random_state)
            train_val_list = self.train_dict["image_id"].values.tolist()
            train_val_labels = np.stack(self.train_dict["image_label_vec"].values, axis=0)

            print("Creating train-val splits...")
            train_list, _, val_list, _ = iterative_train_test_split(np.array(train_val_list)[:, np.newaxis], train_val_labels, test_size=0.1)
            train_list = list(train_list.flatten())
            val_list = list(val_list.flatten())
            for fname, data in zip(['train_split.txt', 'val_split.txt'], [train_list, val_list]):
                with open(os.path.join(self.metadata_dir, fname), 'w') as f:
                    for line in data:
                        f.write(f"{line}\n")
        
        # Now, just train list
        train_list = pd.read_csv(os.path.join(self.metadata_dir, 'train_split.txt'), header=None)
        train_list = train_list[0].copy().values
        train_labels = np.stack(self.train_dict[self.train_dict["image_id"].isin(train_list)]["image_label_vec"].values, axis=0)

        if self.tr_perc < 1.0 and self.tr_perc > 0.0:
            # Check if train-val split exists
            if not os.path.exists(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt')):
                # Generate fixed train split (to ensure they are always the same since the iterative_train_test_split doesn't have a random_state)
                print(f"Creating train_{self.tr_perc} splits...")
                train_list, _, _, _ = iterative_train_test_split(np.array(train_list)[:, np.newaxis], train_labels, test_size=1.0-self.tr_perc)
                train_list = list(train_list.flatten())
                with open(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt'), 'w') as f:
                    for line in train_list:
                        f.write(f"{line}\n")
            
            # Get train list
            train_list = pd.read_csv(os.path.join(self.metadata_dir, f'train_{self.tr_perc}_split.txt'), header=None)
            train_list = train_list[0].copy().values

        # Get validation list
        val_list = pd.read_csv(os.path.join(self.metadata_dir, 'val_split.txt'), header=None)
        val_list = val_list[0].copy().values

        # Get test list
        test_list = list(self.test_dict.copy().keys())

        return train_list.copy(), val_list.copy(), test_list.copy()


    # Method: __len__
    def __len__(self):
        return len(self.data_dict)
    

    # Method: __getitem__
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_dict.iloc[idx]
        
        # Get image id
        image_id = row["image_id"]
        
        # Get labels
        label = row["image_label_vec"]

        # Open image
        image = cv2.imread(os.path.join(self.images_dir, self.subset.replace("validation", "train"), f"{image_id}.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        image /= 255.

        # Apply image transforms
        if self.transform:
            bbox = []
            if self.return_bbox:
                # ignore discrepancies where some radiologists tagged bounding boxes and others did not tag anything
                bbox = [np.array(itm) for itm in row["bbox"] if (not np.all(np.isnan(itm)))]
                cnames = [x for x,y in zip(row["class_name"], row["bbox"]) if (not np.all(np.isnan(y)))]
                if len(bbox) == 0: bbox = None
            else: 
                bbox = None   
            image, bbox = self.transform(image, bbox)
            if bbox:
                bbox = [np.array(x) for x in bbox]
                
        # Convert label to tensor
        label = torch.as_tensor(label).long()

        if self.return_id:
            if self.return_bbox: return image, label, bbox if bbox != None else [], cnames if bbox != None else [], image_id
            else: return image, label, image_id
        
        if self.return_bbox:
            return image, label, bbox if bbox != None else [], cnames if bbox != None else []
        return image, label

# Example usage
if __name__ == "__main__":
    from bcos.data.presets import VinDrCXRClassificationPresetTrain

    vin = VinDrCXRDataset(
        images_dir="/media/TOSHIBA6T/ICRTO/DATASETS/VINDRCXR/images-resized",
        metadata_dir="/media/TOSHIBA6T/ICRTO/DATASETS/VINDRCXR/annotations-resized",
        subset="train",
        transform=VinDrCXRClassificationPresetTrain()
    )
    vin.return_bbox = True
    vin.return_id = True

    disc = 0
    nofind = 0

    def has_discrepancy(lst):
        nan_counts = np.sum(np.isnan(lst), axis=1)
        return np.any(nan_counts > 0) and np.any(nan_counts < 4)
    
    for img, label, bbox, imgid in tqdm(vin):
        if bbox:
            if has_discrepancy(bbox): 
                print(bbox) 
                disc += 1
        if bbox is None or np.sum(np.array(label)) == 0:
            if bbox is not None: print(bbox)
            nofind += 1
    
    print(disc, nofind)
