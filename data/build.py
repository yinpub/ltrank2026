
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



class BUSIDataset(Dataset):
    """
    BUSI Breast Ultrasound dataset (classification only)
    Folder structure:
        root/
            benign/
            malignant/
            normal/
    """

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        train_ratio=0.6,
        val_ratio=0.15,
        random_state=42
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        class_to_label = {
            "normal": 0,
            "benign": 1,
            "malignant": 2,
        }


        image_paths = []
        labels = []

        # -----------------------------
        # 1. all file
        # -----------------------------
        for class_name, label in class_to_label.items():

            class_dir = os.path.join(root_dir, class_name)

            for fname in os.listdir(class_dir):

                #  mask filter
                if "_mask" in fname:
                    continue

                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # -----------------------------
        # 2. train / temp split
        # -----------------------------
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=random_state
        )

        # -----------------------------
        # 3. val / test split
        # -----------------------------
        val_size = val_ratio / (1 - train_ratio)

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=random_state
        )

        # -----------------------------
        # 4. choose mode
        # -----------------------------
        if split == "train":
            self.image_paths = train_paths
            self.labels = train_labels
        elif split == "val":
            self.image_paths = val_paths
            self.labels = val_labels
        elif split == "test":
            self.image_paths = test_paths
            self.labels = test_labels
        else:
            raise ValueError(f"Unknown split: {split}")

        # -----------------------------
        # 5. class info
        # -----------------------------
        self.num_classes = 3
        self.class_names = ["0", "1", "2"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(dataset_name, root_dir, transform_train, transform_test):

    dataset_name = dataset_name.lower()
    if dataset_name == "busi":
        train_dataset = BUSIDataset(root_dir, split="train", transform=transform_train)
        val_dataset   = BUSIDataset(root_dir, split="val", transform=transform_test)
        test_dataset  = BUSIDataset(root_dir, split="test", transform=transform_test)
    return train_dataset, val_dataset, test_dataset

