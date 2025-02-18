from collections import defaultdict
import json
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset  # Assuming PyTorch, for now
from tqdm import tqdm
from loguru import logger as LOGGER

class BDDObjectDetectionDataset(Dataset):
    """
    BDD100K Object Detection Dataset.
    """
    def __init__(self, data_root, split='train', use_cache=True):
        """
        Initializes the BDD100K Object Detection Dataset.

        data_root (str): Path to the root directory of the BDD100K dataset.
        split (str): Split of the dataset to load ('train', 'val', 'test').
        use_cache (bool): Whether to cache the annotations to a parquet file for faster loading.
        """
        self.data_root = data_root
        self.split = split
        self.image_dir = os.path.join(data_root, 'bdd100k_images_100k', 'bdd100k', 'images', '100k', split)
        self.labels_path = os.path.join(data_root, 'bdd100k_labels_release', 'bdd100k', 'labels', f'bdd100k_labels_images_{split}.json')
        self.data_df = None
        if use_cache:
            if os.path.exists(f'bdd100k_{split}_cache.parquet'):
                LOGGER.info(f'Loading {split} annotations from cache.')
                self.data_df = pd.read_parquet(f'bdd100k_{split}_cache.parquet')
            else:
                LOGGER.info(f'Cache not found. Loading {split} annotations from JSON.')
                self._load_annotations()
                self.data_df.to_parquet(f'bdd100k_{split}_cache.parquet')
                LOGGER.info(f'Cached {split} annotations to parquet.')
        else:
            LOGGER.info(f'Loading {split} annotations from JSON.')
            self._load_annotations()
        self.image_ids = self.data_df['image_id'].unique()
        self.class_names = self.data_df['category_name'].unique()
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        LOGGER.info(f'Loaded {len(self.image_ids)} images from the {split} split.')
        LOGGER.info(f'Found {len(self.class_names)} unique classes in the {split} split.')

    def _load_annotations(self):
        """
        Loads annotations from the JSON file and creates a DataFrame.
        """
        with open(self.labels_path, 'r') as f:
            labels_data = json.load(f)
        ann_list = []
        for ann_metadata in tqdm(labels_data, desc=f'Loading {self.split} annotations'):
            image_id = ann_metadata['name']
            image_attributes = ann_metadata['attributes']
            timestamp = ann_metadata['timestamp']
            if 'labels' in ann_metadata:
                for label in ann_metadata['labels']:
                    if 'box2d' not in label:
                        continue
                    ann = {
                        'image_id': image_id,
                        'image_attributes': image_attributes,
                        'timestamp': timestamp,
                        'bbox': [label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']],
                        'category_name': label['category'],
                        'ann_id': label['id'],
                        'ann_attributes': label['attributes'],
                        'manual_attributes': label['manualAttributes'],
                        'manual_shape': label['manualShape'],
                    }
                    ann_list.append(ann)
        self.data_df = pd.DataFrame(ann_list)
        self.image_ids = self.data_df['image_id'].unique()
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.data_df['category_name'].unique())}

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns an item (image and target) from the dataset at the given index.

        Returns:
            tuple: (image, target) where target is a dict containing:
                boxes (list[list]): Bounding boxes in [x_min, y_min, x_max, y_max] format.
                labels (list[int]): Class labels for the bounding boxes.
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        target = defaultdict(list)
        target['boxes'] = self.data_df[self.data_df['image_id'] == image_id]['bbox'].tolist()
        target['labels'] = [self.class_to_idx[category_name] for category_name in self.data_df[self.data_df['image_id'] == image_id]['category_name']]

        return image, target


if __name__ == '__main__':
    from config import DATA_ROOT
    dataset = BDDObjectDetectionDataset(DATA_ROOT, split='train')
    image, target = dataset[0]
    print(f"Image shape: {image.size}")
    print(f"Target keys: {target.keys()}")
    if target['boxes']:
        print(f"Sample boxes: {target['boxes'][0]}")
        print(f"Sample label: {dataset.class_names[target['labels'][0]]}")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        plt.imshow(image)
        ax = plt.gca()
        for box_id, box in enumerate(target['boxes']):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            # Add label to the box
            ax.text(box[0], box[1], dataset.class_names[target['labels'][box_id]], color='g')
            ax.add_patch(rect)
        plt.show()
    print("Done!")