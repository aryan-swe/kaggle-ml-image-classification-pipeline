import io
# importing libraries for system operations, randomization, and data manipulation
import os
import random
import threading
import zipfile
from pathlib import Path, PurePosixPath

# importing libraries for data manipulation and visualization
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# defining a class to handle image archives, allowing for reading and processing images from zip files
class ImageArchive:
    def __init__(self, zip_paths, extensions):
        self.zip_paths = tuple(Path(p) for p in zip_paths)
        self.extensions = frozenset(e.lower() for e in extensions) 
        self.members = {}  
        for zip_path in self.zip_paths:
            with zipfile.ZipFile(zip_path) as archive:
                for info in archive.infolist():
                    member = PurePosixPath(info.filename)
                    is_junk = info.is_dir() or member.parts[0] == "__MACOSX" or member.name.startswith("._")
                    if is_junk or member.suffix.lower() not in self.extensions:
                        continue
                    if member.stem in self.members:
                        raise ValueError(f"duplicate image id {member.stem} in {zip_path}")
                    self.members[member.stem] = (zip_path, info.filename, info.CRC, info.file_size)
        self.ids = sorted(self.members)
        self._local = threading.local()

    def __len__(self):
        return len(self.members) # returning the number of valid image files in the archive

    def __contains__(self, image_id):
        return image_id in self.members # checking if a given image ID is present in the archive

    # defining a private method to handle the opening of zip files and caching them for efficient access
    def _handle(self, zip_path):
        state = getattr(self._local, "state", None)
        if state is None or state["pid"] != os.getpid():
            state = {"pid": os.getpid(), "files": {}}
            self._local.state = state
        if zip_path not in state["files"]:
            state["files"][zip_path] = zipfile.ZipFile(zip_path)
        return state["files"][zip_path]

    # defining a method to read the bytes of an image file from the archive based on its image ID
    def read_bytes(self, image_id):
        zip_path, name, _, _ = self.members[image_id]
        return self._handle(zip_path).read(name)

    # defining a method to open an image file from the archive and convert it to RGB format based on its image ID    
    def open_rgb(self, image_id):
        with Image.open(io.BytesIO(self.read_bytes(image_id))) as img:
            return img.convert("RGB")
    # defining a method to retrieve the file size of an image file in the archive based on its image ID
    def file_size(self, image_id):
        return self.members[image_id][3]
    # defining a method to compute a digest for an image file in the archive based on its image ID, which can be used for data integrity checks
    def digest(self, image_id):
        _, _, crc, size = self.members[image_id]
        return f"{crc:08x}-{size}"
    # defining methods to handle the serialization and deserialization of the ImageArchive object for pickling, ensuring that the thread-local state is not included in the serialized state
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_local"]
        return state
    # defining a method to restore the state of the ImageArchive object during unpickling, ensuring that the thread-local state is reinitialized
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

# defining a custom PyTorch Dataset class to handle pet images and their corresponding features, allowing for efficient data loading and transformation 
class PetFeatureDataset(Dataset):
    def __init__(self, frame, archive, class_to_idx, transform=None, id_width=5): # initializing the dataset with a DataFrame containing image IDs and classes, an ImageArchive object, a mapping of class names to indices, an optional transformation function, and the width for zero-padding image IDs
        self.archive = archive
        self.transform = transform
        self.image_ids = [str(i).strip().zfill(id_width) for i in frame["image_id"]] # zero-padding the image IDs to ensure consistent formatting
        missing = [i for i in self.image_ids if i not in archive] # checking for any image IDs that are missing from the archive
        if missing:
            raise FileNotFoundError(f"{len(missing)} image ids not found in archive, e.g. {missing[:5]}") # raising an error if any image IDs are missing from the archive, providing a sample of the missing IDs for debugging
        if "class" in frame.columns:
            self.targets = [class_to_idx[c] for c in frame["class"]] # mapping the class names in the DataFrame to their corresponding indices based on the provided class_to_idx mapping
        else:
            self.targets = [-1] * len(self.image_ids) # assigning a default target value of -1 for images without class labels, indicating that they are unlabeled
    # defining the length method to return the number of images in the dataset
    def __len__(self):
        return len(self.image_ids)

    # defining the method to retrieve an image and its corresponding target based on the provided index, applying any specified transformations to the image before returning it
    def __getitem__(self, index):
        image = self.archive.open_rgb(self.image_ids[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

# defining a function to set the random seed for each worker in a multi-threaded data loading environment, ensuring reproducibility across different workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
