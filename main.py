#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import itk
import numpy as np
import vtk

from src.registration import register_images
from src.segmentation import segment_tumors
from src.analysis import analyze_changes
from src.visualization import visualize_tumor_analysis
from src.utils import save_results

class TumorAnalyzer:
    """
    Main class for longitudinal tumor analysis pipeline.
    
    This class orchestrates the complete analysis workflow:
    1. Image loading and validation
    2. ITK-based registration
    3. Tumor segmentation
    4. Quantitative analysis
    5. VTK visualization
    6. Results saving
    """
    
    def __init__(self, data_folder="Data"):
        """Initialize the tumor analyzer with data folder path."""
        self.data_folder = Path(data_folder)
        self.image1_path = self.data_folder / "case6_gre1.nrrd"
        self.image2_path = self.data_folder / "case6_gre2.nrrd"
        
        # Output paths for visualization
        self.output_dir = Path("output")
        self.brain_file1 = str(self.image1_path)
        self.brain_file2 = str(self.image2_path)
        self.tumor_file1 = str(self.output_dir / "tumor_mask1.nrrd")
        self.tumor_file2 = str(self.output_dir / "tumor_mask2.nrrd")
        
        # ITK image types
        self.PixelType = itk.F
        self.Dimension = 3
        self.ImageType = itk.Image[self.PixelType, self.Dimension]
        
        # Storage for processed images
        self.fixed_image = None
        self.moving_image = None
        self.registered_image = None
        self.tumor_mask1 = None
        self.tumor_mask2 = None
        
        # Analysis results
        self.analysis_results = None
        
    def validate_inputs(self) -> bool:
        """Validate input files and directories."""
        print("Validating input files...")
        
        if not self.data_folder.exists():
            print(f"Data folder not found: {self.data_folder}")
            return False
            
        if not self.image1_path.exists():
            print(f"Image 1 not found: {self.image1_path}")
            return False
            
        if not self.image2_path.exists():
            print(f"Image 2 not found: {self.image2_path}")
            return False
            
        print("Input validation successful")
        return True
        
    def load_images(self) -> bool:
        """Load the two NRRD images."""
        print("Loading medical images...")
        
        try:
            # Load images using ITK
            self.fixed_image = itk.imread(str(self.image1_path), self.PixelType)
            self.moving_image = itk.imread(str(self.image2_path), self.PixelType)
            
            # Validate loaded images
            if self.fixed_image is None or self.moving_image is None:
                print("Failed to load images")
                return False
                
            print(f"Fixed image loaded: {itk.size(self.fixed_image)}")
            print(f"Moving image loaded: {itk.size(self.moving_image)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading images: {e}")
            return False
        
    def register_images(self) -> bool:
        """Register the moving image to the fixed image using ITK."""
        print("Performing image registration...")
        
        try:
            self.registered_image = register_images(self.fixed_image, self.moving_image)
            
            if self.registered_image is None:
                print("Registration failed")
                return False
                
            print("Registration completed successfully")
            return True
            
        except Exception as e:
            print(f"Registration error: {e}")
            return False

    def segment_tumors(self) -> bool:
        """Segment tumors in both images using threshold-based segmentation."""
        print("Segmenting tumors...")
        
        try:
            self.tumor_mask1, self.tumor_mask2 = segment_tumors(
                self.fixed_image, self.registered_image
            )
            
            if self.tumor_mask1 is None or self.tumor_mask2 is None:
                print("Segmentation failed")
                return False
                
            print("Tumor segmentation completed successfully")
            return True
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return False

    def analyze_changes(self) -> bool:
        """Analyze changes between the two tumor masks."""
        print("Analyzing tumor changes...")
        
        try:
            self.analysis_results = analyze_changes(
                self.tumor_mask1, self.tumor_mask2, self.fixed_image
            )
            
            if self.analysis_results is None:
                print("Analysis failed")
                return False
                
            # Print summary
            print("\nAnalysis Summary:")
            print(f"   Dice Coefficient: {self.analysis_results.get('dice_coefficient', 0):.3f}")
            print(f"   Jaccard Index: {self.analysis_results.get('jaccard_index', 0):.3f}")
            print(f"   Volume 1: {self.analysis_results.get('volume1', 0):.0f} mm³")
            print(f"   Volume 2: {self.analysis_results.get('volume2', 0):.0f} mm³")
            print(f"   Volume Change: {self.analysis_results.get('volume_change', 'N/A'):.0f} mm³")
            print(f"   Volume Change %: {self.analysis_results.get('volume_change_percent', 0):.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return False

    def visualize_results(self) -> bool:
        """Visualize the results using VTK."""
        print("Creating 3D visualization...")
        
        try:
            # Ensure output directory exists and files are saved first
            self.output_dir.mkdir(exist_ok=True)
            
            visualize_tumor_analysis(
                self.brain_file1, self.brain_file2, 
                self.tumor_file1, self.tumor_file2, 
                self.analysis_results
            )
            
            print("Visualization completed successfully")
            return True
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return False

    def save_results(self) -> bool:
        """Save analysis results to files."""
        print("Saving results...")
        
        try:
            success = save_results(
                self.registered_image, self.tumor_mask1, 
                self.tumor_mask2, self.analysis_results
            )
            
            if success:
                print("Results saved successfully to output/ directory")
                return True
            else:
                print("Failed to save some results")
                return False
                
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline."""
        print("Starting longitudinal tumor analysis pipeline...")
        
        # Pipeline steps
        steps = [
            ("Input Validation", self.validate_inputs),
            ("Image Loading", self.load_images),
            ("Image Registration", self.register_images),
            ("Tumor Segmentation", self.segment_tumors),
            ("Change Analysis", self.analyze_changes),
            ("Result Saving", self.save_results),
            ("3D Visualization", self.visualize_results),
        ]
        
        for step_name, step_func in steps:
            print(f"\nStep: {step_name}")
            if not step_func():
                print(f"Pipeline failed at: {step_name}")
                return False
        
        print("\nComplete analysis pipeline finished successfully!")
        return True

def main():
    """Main execution function."""
    print("LONGITUDINAL TUMOR ANALYSIS")
    print("=" * 50)
    print("ITK/VTK Mini Project - Medical Image Analysis")
    print("Objective: Analyze tumor evolution between two scans")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TumorAnalyzer()
    
    try:
        # Execute complete analysis pipeline
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\nANALYSIS COMPLETED SUCCESSFULLY!")
            print("Check the 'output/' directory for results")
            print("View the 3D visualization for detailed insights")
        else:
            print("\nANALYSIS FAILED!")
            print("Check error messages above for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your installation and input files")
        sys.exit(1)

if __name__ == "__main__":
    main()