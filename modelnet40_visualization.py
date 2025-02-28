import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_off(file_path):
    """
    Loads a point cloud from an OFF file.
    
    Handles different variations of OFF file formats:
    - Files starting with 'OFF' header
    - Files starting with vertex/face counts
    - Files with empty lines or comments
    """
    with open(file_path, 'r') as f:
        # Skip empty lines and comments
        def next_valid_line():
            while True:
                line = f.readline()
                if not line:  # EOF
                    return None
                line = line.strip()
                if line and not line.startswith('#'):
                    return line

        # Read header
        header = next_valid_line()
        if not header:
            raise ValueError(f"Empty file: {file_path}")

        # Handle case where first line is 'OFF' or contains counts
        if header == 'OFF':
            header = next_valid_line()
        elif header.startswith('OFF'):
            # Some files might have "OFF" and counts on same line
            header = header[3:].strip()
            if not header:
                header = next_valid_line()

        # Parse vertex/face counts
        try:
            counts = header.split()
            num_vertices = int(counts[0])
            # num_faces = int(counts[1])  # We don't use faces
        except (ValueError, IndexError):
            raise ValueError(f"Invalid vertex/face count in file: {file_path}")

        # Read vertices
        vertices = []
        for _ in range(num_vertices):
            line = next_valid_line()
            if not line:
                break
            try:
                vertex = [float(x) for x in line.split()[:3]]  # Only take first 3 values (x,y,z)
                vertices.append(vertex)
            except (ValueError, IndexError):
                print(f"Warning: Skipping invalid vertex in {file_path}")
                continue

        if not vertices:
            raise ValueError(f"No valid vertices found in file: {file_path}")

        return np.array(vertices)

def visualize_point_cloud(points, title="Point Cloud Visualization"):
    """
    Visualizes the point cloud using a 3D scatter plot.
    
    :param points: numpy array of shape (N, 3)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()

class ModelNetOffDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, normalize=True, split='train'):
        """
        Args:
            root_dir (str): Directory containing subdirectories for each class
            num_points (int): Number of points to sample from each mesh
            normalize (bool): Whether to center and scale the point cloud
            split (str): Either 'train' or 'test'
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.normalize = normalize
        self.split = split
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory does not exist: {root_dir}")
            
        self.files = []
        self.labels = []
        
        # List class directories
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if classes:
            print(f"Found {len(classes)} class directories:")
            for cls in classes:
                print(f"- {cls}")
                
            self.classes = sorted(classes)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            for cls in self.classes:
                cls_folder = os.path.join(root_dir, cls, split)  # Add split folder to path
                if not os.path.exists(cls_folder):
                    print(f"Warning: {split} folder not found in {cls}")
                    continue
                    
                # Search for both '.off' and '.OFF' files
                cls_files = glob.glob(os.path.join(cls_folder, '*.off')) + glob.glob(os.path.join(cls_folder, '*.OFF'))
                print(f"Found {len(cls_files)} files in {cls}/{split}")
                
                for file in cls_files:
                    self.files.append(file)
                    self.labels.append(self.class_to_idx[cls])
        else:
            raise ValueError(f"No class directories found in {root_dir}")
        
        if len(self.files) == 0:
            raise ValueError(f"No .off files found in {root_dir} ({split} split). Please check:\n"
                           f"1. The path is correct\n"
                           f"2. The dataset is properly extracted\n"
                           f"3. The files have .off or .OFF extension\n"
                           f"4. The {split} folders exist in class directories")
        
        print(f"Found total of {len(self.files)} OFF files in {root_dir} ({split} split)")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            vertices = load_off(file_path)  # vertices shape: (N, 3)
        except Exception as e:
            print(f"Error loading file: {file_path}")
            raise e
        
        # Randomly sample points (if there are more vertices than num_points)
        if vertices.shape[0] >= self.num_points:
            indices = np.random.choice(vertices.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(vertices.shape[0], self.num_points, replace=True)
        points = vertices[indices, :]
        
        # Normalize: center and scale to unit sphere.
        if self.normalize:
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_distance = np.max(np.linalg.norm(points, axis=1))
            points = points / furthest_distance
        
        # Convert to PyTorch tensor: shape [3, num_points]
        points = torch.from_numpy(points).float().transpose(0, 1)
        label = torch.tensor(self.labels[idx]).long()
        return points, label

def visualize_specific_class(dataset, class_name, num_samples=1):
    """
    Visualize specific number of samples from a given class.
    """
    if class_name not in dataset.classes:
        print(f"Error: '{class_name}' not found. Available classes:")
        for cls in sorted(dataset.classes):
            print(f"- {cls}")
        return

    # Get indices for the specified class
    class_idx = dataset.class_to_idx[class_name]
    class_indices = [i for i, label in enumerate(dataset.labels) if label == class_idx]
    
    if not class_indices:
        print(f"No samples found for class: {class_name}")
        return

    # Randomly sample if num_samples is less than available samples
    selected_indices = np.random.choice(class_indices, min(num_samples, len(class_indices)), replace=False)
    
    for idx in selected_indices:
        points, _ = dataset[idx]
        pts = points.transpose(0, 1).numpy()
        
        # Create new figure for each sample
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o', s=1)
        
        # Set view angle
        ax.view_init(elev=30, azim=45)
        ax.dist = 7
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Class: {class_name}')
        
        plt.show()

def interactive_visualization():
    # Update this path to the location where you extracted ModelNet40
    modelnet40_path = r"C:\Users\joses\Desktop\AI_Final\ModelNet40"
    
    # Create dataset
    dataset = ModelNetOffDataset(
        root_dir=modelnet40_path, 
        num_points=1024, 
        normalize=True,
        split='train'  # Default to train split
    )
    
    while True:
        print("\nModelNet40 Visualization Options:")
        print("1. View all classes (one sample each)")
        print("2. View specific class")
        print("3. View multiple samples from specific class")
        print("4. Switch dataset split (train/test)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Original visualization code for all classes
            num_classes = len(dataset.classes)
            rows = int(np.ceil(np.sqrt(num_classes)))
            cols = int(np.ceil(num_classes / rows))
            
            fig = plt.figure(figsize=(cols * 4, rows * 4))
            plt.subplots_adjust(wspace=0.3, hspace=0.6)
            
            visualized_classes = set()
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            for points, label in dataloader:
                class_idx = label.item()
                class_name = dataset.classes[class_idx]
                
                if class_name in visualized_classes:
                    continue
                    
                visualized_classes.add(class_name)
                ax = fig.add_subplot(rows, cols, len(visualized_classes), projection='3d')
                pts = points[0].transpose(0, 1).numpy()
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o', s=1)
                ax.view_init(elev=30, azim=45)
                ax.dist = 7
                ax.set_title(class_name, fontsize=8)
                ax.set_xlabel('X', fontsize=6)
                ax.set_ylabel('Y', fontsize=6)
                ax.set_zlabel('Z', fontsize=6)
                ax.tick_params(axis='both', which='major', labelsize=6)
                
                if len(visualized_classes) == num_classes:
                    break
            
            plt.suptitle('ModelNet40 Point Cloud Samples (One per Class)', fontsize=16)
            plt.show()
            
        elif choice == '2':
            print("\nAvailable classes:")
            for cls in sorted(dataset.classes):
                print(f"- {cls}")
            class_name = input("\nEnter class name: ").strip().lower()
            visualize_specific_class(dataset, class_name, num_samples=1)
            
        elif choice == '3':
            print("\nAvailable classes:")
            for cls in sorted(dataset.classes):
                print(f"- {cls}")
            class_name = input("\nEnter class name: ").strip().lower()
            try:
                num_samples = int(input("Enter number of samples to view: "))
                visualize_specific_class(dataset, class_name, num_samples=num_samples)
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '4':
            current_split = dataset.split
            new_split = 'test' if current_split == 'train' else 'train'
            dataset = ModelNetOffDataset(
                root_dir=modelnet40_path,
                num_points=1024,
                normalize=True,
                split=new_split
            )
            print(f"Switched to {new_split} split")
            
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    interactive_visualization()
