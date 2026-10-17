import json
import os

import kagglehub
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def cargar_modelo(nombre):
    if nombre == "hf":
        return YOLO(hf_hub_download(repo_id="melihuzunoglu/human-fall-detection", filename="best.pt"))
    if nombre == "camthink":
        return YOLO("/home/juanchx/Documents/Trabajo/Modelo_caidas/Modelos/yolov8-fall-detection-25/4a7eba2e982955d7c5f2ea65968f23332c123275116e75e2bac915f40ba749e0.pt")
    if nombre == "kaggle":
        path = kagglehub.model_download("bertnardomariouskono/fall-detection-yolov8/pyTorch/yolov8-nano-best-weights")
        return YOLO(f"{path}/{os.listdir(path)[0]}")
    raise ValueError(f"modelo desconocido: {nombre}")


class Dataset:
    def __init__(self, ruta="datasets.json"):
        self.datos = json.load(open(ruta))

    def nombres(self):
        return sorted({n for bloque in self.datos.values() for n in bloque})

    def clases(self, nombre):
        return [b for b, ds in self.datos.items() if nombre in ds]

    def tipo(self, nombre):
        return next(ds[nombre]["tipo"] for ds in self.datos.values() if nombre in ds)

    def carpetas(self, nombre, clase):
        return list(self.datos[clase][nombre]["carpetas"])


def probar(model, path, n=5, conf=0.8):
    resultados = {}
    for video in os.listdir(path):
        predicciones_fin = dict(zip(model.names.keys(), [0]*len(model.names)))
        dict_perdicciones = dict(zip(model.names.keys(), [0]*len(model.names)))
        last_class = dict(zip(model.names.keys(), [0]*len(model.names)))
        Pred = model.predict(source=path + '/' + video, conf=conf, save=False, verbose=False, stream=True)
        for prediction in Pred:
            pred_frame_actual = dict(zip(model.names.keys(), [0]*len(model.names)))
            for valor in dict_perdicciones.keys():
                for clase in prediction.boxes.cls.tolist():
                    if valor == clase:
                        pred_frame_actual[valor] = 1
            for l_class in [key for key in last_class.keys() if last_class[key] > 0]:
                if l_class in [key for key in pred_frame_actual.keys() if pred_frame_actual[key] > 0]:
                    dict_perdicciones[l_class] += 1
                if l_class in [key for key in pred_frame_actual.keys() if pred_frame_actual[key] == 0]:
                    dict_perdicciones[l_class] = 0
            for key, value in dict_perdicciones.items():
                if value == n:
                    predicciones_fin[key] += 1
                    dict_perdicciones[key] = 0
            last_class = pred_frame_actual
        resultados[video] = {model.names[k]: v for k, v in predicciones_fin.items()}
    return resultados


def probar_imagenes(model, path, conf=0.8):
    resultados = {}
    for img in os.listdir(path):
        r = model.predict(source=path + '/' + img, conf=conf, verbose=False)[0]
        clases = [model.names[int(c)] for c in r.boxes.cls]
        resultados[img] = {c: clases.count(c) for c in model.names.values()}
    return resultados


def correr(model, ds, dataset, clase, n=4, conf=0.8):
    resultados = {}
    for carpeta in ds.carpetas(dataset, clase):
        if ds.tipo(dataset) == "imagen":
            res = probar_imagenes(model, carpeta, conf)
        else:
            res = probar(model, carpeta, n, conf)
        for archivo, r in res.items():
            resultados[f"{carpeta}/{archivo}"] = r
    return resultados


clase_caida = {"hf": "fallen", "camthink": "Fall-Detected", "kaggle": "Fall Detected"}


def resumen(res, clase):
    total = len(res)
    con = sum(1 for r in res.values() if r.get(clase, 0) > 0)
    return {"videos": total, "fall": con, "no_fall": total - con}


def main():
    modelos = {n: cargar_modelo(n) for n in ["hf", "camthink", "kaggle"]}
    ds = Dataset()
    dataset, clase = "GMDCSA24", "fall"

    resultados = {}
    for nombre, model in modelos.items():
        resultados[nombre] = correr(model, ds, dataset, clase, n=4, conf=0.8)

    for nombre, res in resultados.items():
        print(nombre, resumen(res, clase_caida[nombre]))


if __name__ == "__main__":
    main()
