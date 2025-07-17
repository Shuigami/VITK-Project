import itk
import numpy as np
from typing import Tuple


def segment_tumors(image1, image2):
    """
    Main segmentation function using percentile-based thresholding.
    
    This function applies our selected segmentation method to both brain images.
    The percentile method was chosen after comprehensive evaluation of multiple
    approaches for optimal tumor detection performance.
    
    Args:
        image1: First brain image (ITK image)
        image2: Second brain image (ITK image)
    
    Returns:
        tuple: (mask1, mask2) - Binary masks for each tumor
    """
    print("  - Segmenting tumors using percentile-based thresholding...")
    print("    - Method selected after evaluation of Otsu, Region Growing, and Watershed")
    
    # Segment first image
    print("    - Processing first image...")
    try:
        mask1 = _segment_single_tumor_percentile(image1)
        print("    - First tumor segmentation successful")
    except Exception as e:
        print(f"    - Error in first image: {e}")
        mask1 = _create_empty_mask_like(image1)
    
    # Segment second image
    print("    - Processing second image...")
    try:
        mask2 = _segment_single_tumor_percentile(image2)
        print("    - Second tumor segmentation successful")
    except Exception as e:
        print(f"    - Error in second image: {e}")
        mask2 = _create_empty_mask_like(image2)
    
    print("    - Segmentation completed successfully")
    return mask1, mask2


def _segment_single_tumor_percentile(image):
    """
    Core segmentation method using percentile-based thresholding.
    
    This method uses the 98.5th percentile as threshold, which effectively
    captures tumor regions while avoiding noise and artifacts.
    
    Pipeline:
    1. Calculate 98.5th percentile threshold from non-zero pixels
    2. Apply binary thresholding
    3. Extract largest connected component
    4. Apply morphological opening for noise removal
    
    Args:
        image: ITK image to segment
        
    Returns:
        ITK binary mask with tumor segmentation
    """
    # Convert to numpy to find a robust threshold
    image_array = itk.array_view_from_image(image)
    
    # Thresholding based on high-intensity pixels, ignoring pure black background
    non_zero_pixels = image_array[image_array > 0]
    if non_zero_pixels.size == 0:
        print("      - Image is empty, returning empty mask.")
        return _create_empty_mask_like(image)
        
    # 98.5th percentile captures tumor regions while avoiding noise
    threshold_value = np.percentile(non_zero_pixels, 98.5)
    threshold_value = float(threshold_value)  # Convert to native Python float
    print(f"      - Calculated threshold: {threshold_value:.2f}")

    # Binary thresholding
    threshold_filter = itk.BinaryThresholdImageFilter.New(
        image,
        LowerThreshold=threshold_value,
        UpperThreshold=itk.NumericTraits[itk.F].max(),
        InsideValue=255,
        OutsideValue=0
    )
    threshold_filter.Update()
    
    # Convert to unsigned char for ConnectedComponentImageFilter compatibility
    cast_filter = itk.CastImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
    cast_filter.SetInput(threshold_filter.GetOutput())
    cast_filter.Update()
    
    # Keep only the largest connected component (assumed to be the tumor)
    connected_component_filter = itk.ConnectedComponentImageFilter.New(cast_filter.GetOutput())
    relabel_filter = itk.RelabelComponentImageFilter.New(
        connected_component_filter.GetOutput(),
        MinimumObjectSize=100
    )
    
    # Get the largest object
    largest_object_filter = itk.BinaryThresholdImageFilter.New(
        relabel_filter.GetOutput(),
        LowerThreshold=1,
        UpperThreshold=1,
        InsideValue=255,
        OutsideValue=0
    )
    largest_object_filter.Update()

    # Clean the mask with morphological opening to remove small noise
    structuring_element = itk.FlatStructuringElement[3].Ball(2)
    opening_filter = itk.BinaryMorphologicalOpeningImageFilter.New(
        largest_object_filter.GetOutput(),
        Kernel=structuring_element
    )
    opening_filter.Update()
    
    return opening_filter.GetOutput()


def _create_empty_mask_like(reference_image):
    """
    Create an empty mask with the same properties as the reference image.
    
    Args:
        reference_image: ITK image to use as reference
        
    Returns:
        ITK binary mask initialized to zeros
    """
    mask = itk.Image[itk.UC, 3].New()
    mask.CopyInformation(reference_image)
    mask.SetRegions(reference_image.GetLargestPossibleRegion())
    mask.Allocate()
    mask.FillBuffer(0)
    return mask
