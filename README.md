# Descarga y prueba del modelo Kraken `kraken-ppocrv6-small`

Modelo de reconocimiento de texto (HTR/OCR) para Kraken, de la familia PP-OCRv6
(variantes tiny, small y medium), publicado por Benjamin Kiessling (ALMAnaCH, Inria).

- Ficha en Hugging Face: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-small
- Tamaño: unos 3,24 millones de parámetros
- Licencia: Apache-2.0
- Requiere: `kraken >= 7.1.0`

## 0. Idea clave: todo en el mismo entorno

Kraken, OpenCV y `huggingface_hub` tienen que estar instalados **en el mismo entorno**
que usas para ejecutar los scripts. Si instalas Kraken en un sitio y OpenCV en otro,
saldrán errores como `ModuleNotFoundError: No module named 'cv2'`.

Conda no es obligatorio (también vale `venv`, ver más abajo), pero es una forma cómoda
de tener un entorno aislado para el proyecto, y es lo que se describe aquí.

## 1. Instalar conda (Miniconda) en WSL

Si ya tienes conda (`conda --version` imprime una versión), salta al apartado 2.

```shell

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

```

- Pulsa Enter para avanzar y escribe `yes` para aceptar la licencia.
- Cuando pregunte si quieres inicializar conda (`conda init`), responde `yes`.
- **Cierra la terminal y ábrela de nuevo** para que el cambio surta efecto.

Comprueba que funciona:

```shell

conda --version

```
## 2. Crear el entorno del proyecto e instalar todo dentro

```shell

conda create -n proyecto_vision python=3.11 
# esta orden te pregunta si quieres instalar una serie de paquetes de primeras, acepta.

conda activate proyecto_vision
python -m pip install "kraken>=7.1.0" huggingface_hub opencv-python

```


- `proyecto_vision` es el nombre del entorno; puedes poner otro.
- Python 3.11 es una elección razonable, pero no he verificado qué versiones de Python
  admite Kraken 7.1. Si `pip` se queja de la versión, prueba con 3.10 o 3.12.
- Después de activar el entorno, el nombre aparece entre paréntesis al principio de la
  línea de la terminal: `(proyecto_vision) usuario@equipo:~$`.

Comprueba que todo quedó instalado en este entorno:

```shell

python -c "import cv2; print(cv2.__version__)"
pip show kraken

```

## 3. Volver a usar el entorno cuando cierras y abres la terminal

Cada vez que abras una terminal nueva hay que **activar el entorno de nuevo**; no se
queda activado solo:

```shell

conda activate kraken-env

```

Comandos útiles:

```shell

conda env list          # ver los entornos que tienes (el activo lleva un *)
conda deactivate        # salir del entorno

```

Si al escribir `conda activate` sale `conda: command not found`, conda no se inicializó:

```shell

  ~/miniconda3/bin/conda init bash

```


Cierra la terminal, ábrela de nuevo e inténtalo otra vez.

Si prefieres que no se active el entorno `base` automáticamente al abrir la terminal:

```shell
conda config --set auto_activate_base false
```

## 5. Descargar el modelo

Con el entorno activado:

```shell

hf download small-models-for-glam/kraken-ppocrv6-small small.safetensors --local-dir ./kraken-ppocrv6-small

```

El fichero quedará en `./kraken-ppocrv6-small/small.safetensors`.


## Problemas conocidos

- **`ModuleNotFoundError: No module named 'cv2'`** (o de otro paquete): el entorno
  activo no es el que tiene los paquetes instalados. Comprueba que el nombre del
  entorno aparece entre paréntesis en la terminal y, si no, ejecuta
  `conda activate kraken-env`.
- **`Segmentation.__init__() missing ... 'script_detection'`**: en `a_segmentacion_kraken`
  (`lineas.py`) añade `script_detection=False` a la llamada a `Segmentation(...)`.
- **La firma de `modelo.predict(...)` no está confirmada** para la versión 7.1 de
  Kraken. Si da error, es el primer sitio donde mirar.
- **Acentos descompuestos**: el modelo se entrenó con texto en Unicode NFD, así que una
  letra con tilde puede salir como letra + acento suelto. Se corrige con:
  `unicodedata.normalize("NFC", texto)`.
- **Nombre del fichero**: asegúrate de que se llama `lineas.py` (en minúsculas), porque
  el código lo importa como `lineas`.

## Limitaciones del modelo a tener en cuenta

- El español de entrenamiento es sobre todo histórico (manuscritos medievales y del
  siglo XIX, pliegos de cordel impresos), no letra moderna. La ficha da un CER del 5,00 %
  y un WER del 18,82 % para español, pero el test no separa documentos, así que mide
  generalización dentro del dominio. Con letra moderna se espera peor resultado, y ahí
  entra el fine-tuning con líneas propias.
- Es un modelo base pensado para poder hacerle fine-tuning.
- Se entrenó con línea base y polígono por línea, no con cajas; con cajas puede leer algo peor.

## Fuentes

- Ficha del modelo: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-small
- Introducción a ATR en Kraken: https://kraken.re/main/introduction_to_atr.html
- Documentación de la API de Kraken: https://kraken.re/7.1/_sources/user_guide/api.rst.txt