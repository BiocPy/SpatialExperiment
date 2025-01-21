import biocframe


def _validate_sample_image_ids(img_data: biocframe.BiocFrame, new_sample_id: str, new_image_id: str):
    for row in img_data:
        data = row[1]
        if data["sample_id"] == new_sample_id and data["image_id"] == new_image_id:
            raise ValueError(f"Image with Sample ID: {new_sample_id} and Image ID: {new_image_id} already exists")

def _validate_spatial_coords(spatial_coords, shape):
    if spatial_coords is None:
        return
    
    if not isinstance(spatial_coords, biocframe.BiocFrame):
        raise TypeError("'spatial_coords' is not a BiocFrame object.")
    
    if shape[1] != spatial_coords.shape[0]:
        raise ValueError(f"Spatial coordinates do not contain coordinates for all cells.")
