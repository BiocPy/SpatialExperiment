from typing import Union
import biocframe


def retrieve_rows_by_id(
    img_data: biocframe.BiocFrame,
    sample_id: Union[str, True, None] = None,
    image_id: Union[str, True, None] = None,
) -> biocframe.BiocFrame:
    if sample_id is True:
        if image_id is True:
            return img_data

        elif image_id is None:
            unique_sample_ids = list(set(img_data["sample_id"]))
            sample_id_groups = img_data.split("sample_id")
            subset = None

            # get the first entry for all samples
            for sample_id in unique_sample_ids:
                row = sample_id_groups[sample_id][0, :]
                if subset is None:
                    subset = row
                else:
                    subset = subset.combine_rows(row)
        else:
            subset = img_data[
                [_image_id == image_id for _image_id in img_data["image_id"]]
            ]

    elif sample_id is None:

        first_sample_id = img_data["sample_id"][0]
        first_sample = img_data[
            [_sample_id == first_sample_id for _sample_id in img_data["sample_id"]]
        ]

        if image_id is True:
            # get all entries for the first sample
            subset = first_sample

        elif image_id is None:
            # get the first entry
            subset = first_sample[0, :]
        else:
            subset = first_sample[
                [_image_id == image_id for _image_id in img_data["image_id"]]
            ]

    else:
        selected_sample = img_data[
            [_sample_id == sample_id for _sample_id in img_data["sample_id"]]
        ]
        if image_id is True:
            subset = selected_sample
        elif image_id is None:
            subset = selected_sample[0, :]
        else:
            subset = selected_sample[
                [_image_id == image_id for _image_id in selected_sample["image_id"]]
            ]

    return subset
