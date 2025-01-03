import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class MVTecAnalyzer:
    def __init__(self, dataset_path=None):
        """
        Initialize MVTec dataset analyzer
        
        Args:
            dataset_path (str): Path to the existing MVTec dataset location
                              If None, uses default Kaggle cache location
        """
        # Use provided path or default to Kaggle cache location
        self.dataset_path = dataset_path or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "kagglehub",
            "datasets",
            "ipythonx",
            "mvtec-ad",
            "versions",
            "2"
        )
        self.stats = None

    def verify_dataset(self):
        """
        Verify the dataset structure and integrity.
        Returns True if at least one category is found.
        """
        print(f"Verifying dataset structure in {self.dataset_path}...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found at {self.dataset_path}")
            
        expected_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
        
        found_categories = []
        missing_categories = []
        
        # Check each expected category
        for category in expected_categories:
            category_path = os.path.join(self.dataset_path, category)
            
            if not os.path.exists(category_path):
                missing_categories.append(category)
                continue
                
            found_categories.append(category)
            
            # Verify train/test split directories
            for split in ['train', 'test']:
                split_path = os.path.join(category_path, split)
                if not os.path.exists(split_path):
                    print(f"Warning: {split} directory missing in {category}!")
        
        print("\nDataset verification results:")
        print(f"Found categories ({len(found_categories)}): {', '.join(found_categories)}")
        if missing_categories:
            print(f"Missing categories ({len(missing_categories)}): {', '.join(missing_categories)}")
        
        return len(found_categories) > 0

    def get_dataset_stats(self):
        """
        Collect comprehensive statistics about the dataset.
        Returns a dictionary containing statistics for each category.
        """
        print("\nCollecting dataset statistics...")
        dataset_stats = {}
        
        # Get list of actual categories in the dataset
        categories = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # Process each category
        for category in categories:
            category_path = os.path.join(self.dataset_path, category)
            
            # Initialize counters
            train_good = test_good = test_defective = 0
            defect_types = []
            
            # Count training images (good samples only)
            train_good_path = os.path.join(category_path, "train", "good")
            if os.path.exists(train_good_path):
                train_good = len([f for f in os.listdir(train_good_path) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Process test directory
            test_path = os.path.join(category_path, "test")
            if os.path.exists(test_path):
                # Count good test images
                test_good_path = os.path.join(test_path, "good")
                if os.path.exists(test_good_path):
                    test_good = len([f for f in os.listdir(test_good_path) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                # Count defective test images and identify defect types
                for defect in os.listdir(test_path):
                    if defect != "good":
                        defect_types.append(defect)
                        defect_path = os.path.join(test_path, defect)
                        test_defective += len([f for f in os.listdir(defect_path) 
                                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Store category statistics
            dataset_stats[category] = {
                "train (good)": train_good,
                "test (good)": test_good,
                "test (defective)": test_defective,
                "defective types": len(defect_types),
                "defect_names": defect_types,
                "total": train_good + test_good + test_defective
            }
        
        # Calculate overall totals
        total_stats = {
            "train (good)": sum(stats["train (good)"] for stats in dataset_stats.values()),
            "test (good)": sum(stats["test (good)"] for stats in dataset_stats.values()),
            "test (defective)": sum(stats["test (defective)"] for stats in dataset_stats.values()),
            "defective types": "-",
            "defect_names": "-",
            "total": sum(stats["total"] for stats in dataset_stats.values())
        }
        dataset_stats["TOTAL"] = total_stats
        
        self.stats = dataset_stats
        return dataset_stats

    def plot_distribution(self, save_path=None):
        """
        Plot the distribution of images across categories.
        Creates a figure with a bar plot and a summary table.
        
        Args:
            save_path (str, optional): Path where to save the plot
        """
        if not self.stats:
            print("No statistics available. Run get_dataset_stats first.")
            return
        
        # Prepare plotting data
        categories = [cat for cat in self.stats.keys() if cat != "TOTAL"]
        train_good = [self.stats[cat]["train (good)"] for cat in categories]
        test_good = [self.stats[cat]["test (good)"] for cat in categories]
        test_defective = [self.stats[cat]["test (defective)"] for cat in categories]
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 10))
        
        # Create bar plot
        plt.subplot(2, 1, 1)
        x = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x - width, train_good, width, label='Train (Good)')
        plt.bar(x, test_good, width, label='Test (Good)')
        plt.bar(x + width, test_defective, width, label='Test (Defective)')
        
        plt.xlabel('Categories')
        plt.ylabel('Number of Images')
        plt.title('MVTec Dataset Distribution')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        
        # Add value labels on the bars
        for i, v in enumerate(train_good):
            plt.text(i - width, v, str(v), ha='center', va='bottom')
        for i, v in enumerate(test_good):
            plt.text(i, v, str(v), ha='center', va='bottom')
        for i, v in enumerate(test_defective):
            plt.text(i + width, v, str(v), ha='center', va='bottom')
        
        # Create statistics table
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Prepare table data
        table_data = []
        for cat in categories + ["TOTAL"]:
            row = [
                cat,
                self.stats[cat]["train (good)"],
                self.stats[cat]["test (good)"],
                self.stats[cat]["test (defective)"],
                self.stats[cat]["defective types"],
                self.stats[cat]["total"]
            ]
            table_data.append(row)
        
        columns = ['Category', 'Train (Good)', 'Test (Good)', 
                  'Test (Defective)', 'Defect Types', 'Total']
        
        # Create and style the table
        table = plt.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {save_path}")
        
        plt.show()

def main():
    # Initialize analyzer with default Kaggle cache path
    analyzer = MVTecAnalyzer()
    
    # Run dataset verification
    if analyzer.verify_dataset():
        # Collect and display statistics
        stats = analyzer.get_dataset_stats()
        
        # Display detailed statistics for each category
        print("\nDetailed statistics for each category:")
        for category, stat in stats.items():
            print(f"\n{category}:")
            for key, value in stat.items():
                print(f"  {key}: {value}")
        
        # Create and save visualization
        analyzer.plot_distribution(save_path="mvtec_distribution.png")
    else:
        print("Dataset verification failed!")

if __name__ == "__main__":
    main()