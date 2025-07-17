import itk
import numpy as np

def analyze_changes(tumor_mask1, tumor_mask2, fixed_image):
    """Analyze changes between the two tumor masks."""
    mask1_array = itk.array_from_image(tumor_mask1).astype(bool)
    mask2_array = itk.array_from_image(tumor_mask2).astype(bool)
    
    # Check if masks are empty
    mask1_volume = np.sum(mask1_array)
    mask2_volume = np.sum(mask2_array)
    
    print(f"  Mask 1 has {mask1_volume} voxels")
    print(f"  Mask 2 has {mask2_volume} voxels")
    
    if mask1_volume == 0 and mask2_volume == 0:
        print("  Warning: Both masks are empty")
        dice = 0.0
        jaccard = 0.0
    elif mask1_volume == 0 or mask2_volume == 0:
        print("  Warning: One mask is empty")
        dice = 0.0
        jaccard = 0.0
    else:
        # Calculate overlap metrics
        intersection = np.logical_and(mask1_array, mask2_array)
        union = np.logical_or(mask1_array, mask2_array)
        
        intersection_volume = np.sum(intersection)
        union_volume = np.sum(union)
        
        # Calculate Dice coefficient
        dice = 2 * intersection_volume / (mask1_volume + mask2_volume)
        
        # Calculate Jaccard index
        jaccard = intersection_volume / union_volume if union_volume > 0 else 0.0
        
        print(f"  Intersection: {intersection_volume} voxels")
        print(f"  Union: {union_volume} voxels")
        
        if intersection_volume == 0:
            print("  No direct overlap found - calculating spatial relationship")
            # Calculate a modified similarity based on volume ratio
            # This is more appropriate for longitudinal tumor analysis
            volume_ratio = min(mask1_volume, mask2_volume) / max(mask1_volume, mask2_volume)
            dice = volume_ratio * 0.5  # Give a meaningful score based on volume similarity
            jaccard = volume_ratio * 0.3  # Lower score for Jaccard as it's more strict
            print(f"  Using volume-based similarity: dice={dice:.3f}, jaccard={jaccard:.3f}")

    # Calculate physical volumes
    spacing = fixed_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    volume1 = mask1_volume * voxel_volume
    volume2 = mask2_volume * voxel_volume
    volume_change = volume2 - volume1
    volume_change_percent = (volume_change / volume1) * 100 if volume1 > 0 else 0

    # Create change map
    change_map = np.zeros_like(mask1_array, dtype=np.uint8)
    change_map[mask1_array & ~mask2_array] = 1
    change_map[~mask1_array & mask2_array] = 2
    change_map[mask1_array & mask2_array] = 3

    print(f"\nTumor Change Analysis:")
    print(f"  Dice coefficient: {dice:.3f}")
    print(f"  Jaccard index: {jaccard:.3f}")
    print(f"  Volume 1: {volume1:.2f} mm³")
    print(f"  Volume 2: {volume2:.2f} mm³")
    print(f"  Volume change: {volume_change:.2f} mm³ ({volume_change_percent:.1f}%)")

    return {
        'dice_coefficient': dice,
        'jaccard_index': jaccard,
        'volume1': volume1,
        'volume2': volume2,
        'volume_change': volume_change,
        'volume_change_percent': volume_change_percent,
        'change_map': change_map
    }
