import os
import shutil
import random

def prepare_data(source_dir, train_dir, validation_dir, test_dir, healthy_classes=["Healthy"]):
    """Splits the dataset into train, validation, and test sets for binary classification."""
    # Create directories for binary classification
    for split_dir in [train_dir, validation_dir, test_dir]:
        for cls in ['healthy', 'non_healthy']:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    # Go through the files
    for disease_class in os.listdir(source_dir):
        disease_class_path = os.path.join(source_dir, disease_class)
        if os.path.isdir(disease_class_path):
            images = os.listdir(disease_class_path)
            random.shuffle(images)

            train_split = int(0.8 * len(images))
            val_split = int(0.9 * len(images))

            target_class = "healthy" if disease_class in healthy_classes else "non_healthy"

            for img in images[:train_split]:
                shutil.copy(os.path.join(disease_class_path, img), os.path.join(train_dir, target_class))
            for img in images[train_split:val_split]:
                shutil.copy(os.path.join(disease_class_path, img), os.path.join(validation_dir, target_class))
            for img in images[val_split:]:
                shutil.copy(os.path.join(disease_class_path, img), os.path.join(test_dir, target_class))

if __name__ == "__main__":
    prepare_data(
        source_dir="Lettuce_disease_datasets",
        train_dir="train",
        validation_dir="validation",
        test_dir="test"
    )
