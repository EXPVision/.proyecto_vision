from PIL import Image
import cv2
from kraken.tasks import RecognitionTaskModel
from kraken.configs import RecognitionInferenceConfig
from preprocesado import preprocesar_folio
from Lineas import detectar_lineas, a_segmentacion_kraken

folio = preprocesar_folio("entrada.jpg")
lineas = detectar_lineas(folio.binaria)
cv2.imwrite("folio.png", folio.color)

im = Image.open("folio.png")
seg = a_segmentacion_kraken(lineas, "folio.png")
modelo = RecognitionTaskModel.load_model("./kraken-ppocrv6-small/small.safetensors")

for registro in modelo.predict(im, seg, RecognitionInferenceConfig()):
    print(registro.prediction)
