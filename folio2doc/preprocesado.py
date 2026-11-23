# APLICA FILTROS Y ROTACIONES PARA PREPARAR LA IMAGEN PARA LA DETECCION DE LINEAS

"""Bloque 1 - Paso 1: preprocesado de la foto del folio.

Entrada : foto del folio (ruta o array BGR de OpenCV).
Salida  : un FolioPreprocesado con
            - color    : folio enderezado en BGR (esta es la imagen que se le dara a Kraken)
            - gris     : folio enderezado en escala de grises
            - binaria  : folio enderezado, tinta = 255 y papel = 0 (para detectar lineas)
          Las tres imagenes tienen EXACTAMENTE las mismas dimensiones, asi que una
          coordenada sacada de `binaria` vale tambien para `color`.

Pasos:
    1. Localizar el folio (contorno de 4 esquinas) y aplicar transformacion de perspectiva.
    2. Pasar a escala de grises.
    3. Corregir la iluminacion irregular (sombras, degradados).
    4. Binarizar (Otsu) y limpiar los bordes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Tamano de salida del folio enderezado (A4 a ~200 ppp). Si el folio sale apaisado
# se intercambian ancho y alto.
A4_ANCHO_PX = 1654
A4_ALTO_PX = 2339


@dataclass
class FolioPreprocesado:
    color: np.ndarray
    gris: np.ndarray
    binaria: np.ndarray
    folio_detectado: bool  # False -> no se encontro el folio y se uso la imagen entera
    matriz_perspectiva: np.ndarray | None  # foto original -> folio enderezado


# ---------------------------------------------------------------------------
# 1. Localizar y enderezar el folio
# ---------------------------------------------------------------------------
def _ordenar_esquinas(pts: np.ndarray) -> np.ndarray:
    """Devuelve las 4 esquinas en orden: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    pts = pts.reshape(4, 2).astype(np.float32)
    suma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(suma)], pts[np.argmin(dif)], pts[np.argmax(suma)], pts[np.argmax(dif)]],
        dtype=np.float32,
    )


def _buscar_esquinas_folio(img: np.ndarray, lado_trabajo: int = 1000, area_minima: float = 0.25):
    """Busca el folio asumiendo que el papel es mas claro que lo que hay alrededor.

    Devuelve las 4 esquinas (en coordenadas de la imagen original) o None.
    """
    h, w = img.shape[:2]
    esc = min(1.0, lado_trabajo / max(h, w))
    peq = cv2.resize(img, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA) if esc < 1 else img

    gris = cv2.GaussianBlur(cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)  # rellena la tinta dentro del papel
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)   # quita objetos pequenos claros

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < area_minima * peq.shape[0] * peq.shape[1]:
        return None

    perimetro = cv2.arcLength(mayor, True)
    aprox = cv2.approxPolyDP(mayor, 0.02 * perimetro, True)
    if len(aprox) == 4 and cv2.isContourConvex(aprox):
        esquinas = aprox
    else:  # contorno irregular: usamos el rectangulo rotado minimo del casco convexo
        esquinas = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(mayor)))
    return _ordenar_esquinas(esquinas) / esc


def _enderezar(img: np.ndarray, esquinas: np.ndarray):
    tl, tr, br, bl = esquinas
    ancho = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    alto = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    ancho_s, alto_s = (A4_ANCHO_PX, A4_ALTO_PX) if alto >= ancho else (A4_ALTO_PX, A4_ANCHO_PX)
    destino = np.array([[0, 0], [ancho_s - 1, 0], [ancho_s - 1, alto_s - 1], [0, alto_s - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(esquinas, destino)
    return cv2.warpPerspective(img, M, (ancho_s, alto_s), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)), M


# ---------------------------------------------------------------------------
# 2-4. Gris, iluminacion y binarizacion
# ---------------------------------------------------------------------------
def _normalizar_iluminacion(gris: np.ndarray) -> np.ndarray:
    """Divide la imagen por una estimacion del fondo: el papel queda uniformemente blanco."""
    lado = max(31, (min(gris.shape) // 40) | 1)  # impar; mayor que el grosor de los trazos
    fondo = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (lado, lado)))
    fondo = cv2.GaussianBlur(fondo, (0, 0), lado / 4)
    return cv2.divide(gris, fondo, scale=255)


def _binarizar(gris_norm: np.ndarray, margen_rel: float = 0.015) -> np.ndarray:
    suave = cv2.GaussianBlur(gris_norm, (3, 3), 0)
    _, b = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Los bordes del folio enderezado suelen tener sombras o restos del fondo: se borran.
    m = int(margen_rel * min(b.shape))
    b[:m, :] = 0
    b[-m:, :] = 0
    b[:, :m] = 0
    b[:, -m:] = 0
    return b


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------
def preprocesar_folio(imagen: str | Path | np.ndarray, enderezar: bool = True) -> FolioPreprocesado:
    """Prepara la foto de un folio para segmentar lineas y para pasarsela a Kraken.

    imagen   : ruta a la foto o array BGR ya cargado con cv2.imread.
    enderezar: si False no se busca el folio (util para escaneos ya rectos).
    """
    if isinstance(imagen, (str, Path)):
        img = cv2.imread(str(imagen), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {imagen}")
    else:
        img = imagen.copy()

    detectado, M = False, None
    if enderezar:
        esquinas = _buscar_esquinas_folio(img)
        if esquinas is not None:
            img, M = _enderezar(img, esquinas)
            detectado = True

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaria = _binarizar(_normalizar_iluminacion(gris))
    return FolioPreprocesado(color=img, gris=gris, binaria=binaria, folio_detectado=detectado, matriz_perspectiva=M)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Uso: python preprocesado.py foto.jpg")
    r = preprocesar_folio(sys.argv[1])
    print("Folio detectado:", r.folio_detectado, "| tamano:", r.color.shape[1], "x", r.color.shape[0])
    cv2.imwrite("salida_color.png", r.color)
    cv2.imwrite("salida_binaria.png", r.binaria)
    print("Guardados salida_color.png y salida_binaria.png para revisarlos")
