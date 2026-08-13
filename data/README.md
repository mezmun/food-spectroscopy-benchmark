# Dataset files

Place the following source files in this directory. The data themselves are not redistributed by this repository.

| File | Tasks used by the benchmark |
|---|---|
| `mangos_TA_Vit_C.xlsx` | Mango-A titratable acidity; Mango-A vitamin C |
| `Cucurbitaceae_Fruits.xlsx` | Cucurbitaceae water content; Cucurbitaceae soluble solids/Brix |
| `milk.csv` | Milk fat |
| `Mangoes.xlsx` | Mango-B titratable acidity; vitamin C; Brix |
| `DATASET.csv` | Grape berry sugar |

The row and column slices are defined in `manuscript_datasets()` in `benchmark.py`. The program validates that a target column does not overlap the predictor slice and also checks whether any predictor exactly reproduces the target.
