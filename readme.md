### Modelos usados
1. https://huggingface.co/melihuzunoglu/human-fall-detection

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

model_path = hf_hub_download(repo_id="melihuzunoglu/human-fall-detection", filename="best.pt")

model = YOLO(model_path)
 
results = model.predict(source="image1.jpg", conf=0.25, save=True)
```

2. https://www.camthink.ai/developer-center/models/yolov8-fall-detection-25/
   path: /home/juanchx/Documents/Trabajo/Modelo_caidas/Modelos/yolov8-fall-detection-25


3. yolo8

```python
import kagglehub
# https://www.kaggle.com/models/bertnardomariouskono/fall-detection-yolov8/
# Download latest version
path = kagglehub.model_download("bertnardomariouskono/fall-detection-yolov8/pyTorch/yolov8-nano-best-weights")

print("Path to model files:", path)
```

#############################################
## Dataset

1. Le2i Fall Dataset
    http://kaggle.com/datasets/tuyenldvn/falldataset-imvia?resource=download&select=Coffee_room_02
    # summary 190 videos
    

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("tuyenldvn/falldataset-imvia")

print("Path to dataset files:", path)

```


2. Multiple Cameras Fall Dataset
    https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset
    # summary 192 videos

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumicksarker/multiple-cameras-fall-dataset")

print("Path to dataset files:", path)
```
3. GMDCSA24
  https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos

git clone https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos.git


4. chatGPT
15 ejemplos en imagenes 3 imagenes consecutivas por ejemplo.


5. CCTV Incident Dataset -Fall & Lying Down Detection
    https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection?select=laying_dataset

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("simuletic/cctv-incident-dataset-fall-and-lying-down-detection")

print("Path to dataset files:", path)
```


# Dataset actividades normales

1. Indoor Action Dataset
    https://github.com/DaniDeniz/IndoorActionDataset/tree/main

| Action name | Train | Val. | Test | Total |
|-------------|-------|------|------|-------|
| blowing nose or sneezing | 38 | 10 | 15 | 63 |
| cleaning | 55 | 17 | 34 | 106 |
| eating | 57 | 18 | 32 | 107 |
| falling down | 34 | 11 | 16 | 61 |
| lying on the floor | 53 | 12 | 36 | 101 |
| sitting down | 66 | 22 | 29 | 117 |
| standing up | 95 | 30 | 41 | 166 |
| walking | 158 | 34 | 90 | 282 |
| watching tv | 60 | 19 | 43 | 122 |
| no-action | 57 | 14 | 32 | 103 |

2. OFFICEDATA
    https://homepages.inf.ed.ac.uk/rbf/OFFICEDATA/
  20 dias 2 oficinas (solo nos descargamos 5 dias )
       0 --- Room is empty (the position values are also 0)
       1 --- Person is standing/walking
       2 --- Person is sitting
       3 --- Two or three people are talking to each other
       4 --- Person in room has fallen


# datasets.json

Indice de las rutas de los datasets para probar los modelos. Estructura: `bloque -> dataset -> {tipo, carpetas: {ruta: numero de archivos}}`.

- `fall`: datasets (o carpetas) solo con caidas.
- `no_fall`: datasets (o carpetas) solo con actividades normales.
- `fall_adl`: datasets mezclados (caidas y actividades normales en las mismas carpetas), por ejemplo Le2i.
- `tipo`: `video` o `imagen`.

Rutas relativas a `Modelo_caidas`, salvo las de la cache de kagglehub, que son absolutas.
