# Bibliotecas generales
import cv2 as cv
import argparse
import json
from pathlib import Path
from PIL import Image

# Modelo: Kraken
from kraken.tasks import RecognitionTaskModel
from kraken.configs import RecognitionInferenceConfig

# Nuestros .py
from preprocesado import preprocesar_folio
from lineas import detectar_lineas, a_segmentacion_kraken, dibujar_lineas

# Buscar el path
BASE = Path(__file__).resolve().parent

# Argumentos de la imagen
parser = argparse.ArgumentParser(description="Reconocer texto manuscrito con Kraken.")
parser.add_argument("--imagen", "-i", default=str(BASE / "entrada.jpg"))
args = parser.parse_args()

# Leer la ruta personal del modelo
try:
    with open(BASE / "config" / "config.json", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    raise SystemExit("Falta config/config.json. Créalo a partir de config.example.json")

ruta_modelo = Path(config["modelo"]).expanduser()

if not ruta_modelo.is_file():
    raise SystemExit(f"No se encontró el modelo en: {ruta_modelo}")

# Cargar la imagen y el modelo
folio = preprocesar_folio(args.imagen)
lineas = detectar_lineas(folio.binaria)

# Salir si no se detectan líneas
if not lineas:
    raise SystemExit("No se han detectado líneas de texto en la imagen.")

# Guardar imágenes de procesos intermedios (luego podremos eliminarlo)
ruta_resultados = BASE / "resultados"
ruta_resultados.mkdir(exist_ok=True)

cv.imwrite(str(ruta_resultados / "folio_color.png"), folio.color)
cv.imwrite(str(ruta_resultados / "folio_binario.png"), folio.binaria)
cv.imwrite(str(ruta_resultados / "lineas_detectadas.png"), dibujar_lineas(folio.color, lineas))

# Pasar a RGB para PIL
im = Image.fromarray(cv.cvtColor(folio.color, cv.COLOR_BGR2RGB))

seg = a_segmentacion_kraken(lineas, "folio.png")
modelo = RecognitionTaskModel.load_model(str(ruta_modelo))

for registro in modelo.predict(im, seg, RecognitionInferenceConfig()):
    print(registro.prediction)
