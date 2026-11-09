# Data

Neither dataset is redistributed here.

## NEU-Seg (surface defects, pixel annotations)

Mendeley Data, CC BY 4.0: <https://doi.org/10.17632/z8g47vzt4p.1>

The release used here contains one labelled split. After extraction the tree is:

```
data/NEU_Seg-main/
  images/test/*.jpg
  annotations/test/*.png      # class ids 1..3, background 0
```

`NEUSEG_ROOT` must point at `data/NEU_Seg-main`.

## STM point defects (Si / Ge / TiO2)

Zenodo, CC BY 4.0: <https://doi.org/10.5281/zenodo.15525935> (file `Data.zip`)

After extraction:

```
data/STM/Segmentation_data/
  Ge(001)-AsH3/{images,masks}/*.npy
  Si(001)-H-AsH3/{images,masks}/*.npy
  TiO2(110)/{images,masks}/*.npy
```

Masks are binary and stored at half the image resolution; `src/stm_data.py`
upsamples them with nearest-neighbour interpolation. Mask filenames end in either
`_mask.npy` or `_mask_full.npy`; both are handled.

`STM_ROOT` must point at `data/STM/Segmentation_data`.

## Licence

Datasets keep their own licences (CC BY 4.0). The code in this repository is MIT.
