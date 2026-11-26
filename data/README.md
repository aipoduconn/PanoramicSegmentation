# Data layout

No data is included in this repository: the panoramic radiographs and their annotations are patient data. To run the pipeline on your own data, arrange it as follows (`<set_name>` is any name for a batch of images, e.g. `pan_set_1`):

```
data/
├── raw_imgs/<set_name>/            # Radiograph images (.jpg/.png)
├── annots/
│   ├── manual_download/<set_name>.json   # COCO annotation file exported from COCO Annotator
│   └── code_moved/                 # Processed JSON files are moved here automatically
├── mask/<set_name>/                # Generated: label masks (<image file name>.png)
├── augmented/
│   ├── img/                        # Generated: augmented images
│   └── mask/                       # Generated: augmented masks
└── metadata/                       # Generated: train_data.csv, test_data.csv
```

Create the folders for the generated augmented images before you run preprocessing:

```bash
mkdir -p data/annots/manual_download data/annots/code_moved data/augmented/img data/augmented/mask
```

## Annotation format

- One COCO JSON file per image set. Its `file_name` entries must match the file names in `data/raw_imgs/<set_name>/`.
- Each tooth is a polygon `segmentation` with one of the 32 category names `UR1`–`UR8`, `UL1`–`UL8`, `LL1`–`LL8`, `LR1`–`LR8`.
- Only images marked `annotated: true` are used.
- The mapping from category IDs to class indices (`ID2IDX` in `src/data/utils.py`) matches the COCO Annotator project used in the study. If your category IDs differ, update it.
