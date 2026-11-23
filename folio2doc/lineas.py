# SEGMENTA LAS LINEAS Y OBTIENE LAS ESQUENAS DE LAS BBOX PARA PASAR POR ENTRADA AL MODELO
"""Bloque 1 - Paso 2: deteccion de lineas de texto y su posicion en el folio.

Entrada : imagen binaria del folio enderezado (tinta = 255), sacada de preprocesado.py
Salida  : lista de `Linea`, ordenadas de arriba abajo (orden de lectura), cada una con
            - bbox       : (x0, y0, x1, y1) en pixeles del folio enderezado
            - bbox_norm  : lo mismo normalizado entre 0 y 1 (no depende de la resolucion)

Metodo clasico: se dilata la tinta en horizontal para unir las palabras de cada linea,
se calcula el perfil de proyeccion horizontal (cuanta tinta hay en cada fila) y los
huecos entre picos separan las lineas.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Linea:
    id: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (px)
    bbox_norm: tuple[float, float, float, float]  # x0, y0, x1, y1 en [0, 1]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _limpiar_ruido(tinta: np.ndarray, area_min: int = 20) -> np.ndarray:
    """Elimina manchas diminutas (polvo, ruido de la binarizacion)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    validos = np.zeros(n, dtype=bool)
    validos[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min
    return np.where(validos[etiquetas], 255, 0).astype(np.uint8)


def _quitar_renglones(tinta: np.ndarray) -> np.ndarray:
    """Quita lineas horizontales largas (papel rayado). Opcional: puede comerse algo de trazo."""
    ancho = tinta.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(0.35 * ancho)), 1))
    renglones = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k)
    renglones = cv2.dilate(renglones, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    return cv2.subtract(tinta, renglones)


def _bandas_de_filas(perfil: np.ndarray, umbral: float, hueco_min: int) -> list[tuple[int, int]]:
    """Tramos consecutivos de filas con perfil > umbral; une tramos separados por huecos pequenos."""
    activo = perfil > umbral
    bandas, inicio = [], None
    for y, a in enumerate(activo):
        if a and inicio is None:
            inicio = y
        elif not a and inicio is not None:
            bandas.append([inicio, y])
            inicio = None
    if inicio is not None:
        bandas.append([inicio, len(activo)])

    unidas: list[list[int]] = []
    for b in bandas:
        if unidas and b[0] - unidas[-1][1] < hueco_min:
            unidas[-1][1] = b[1]
        else:
            unidas.append(b)
    return [(a, b) for a, b in unidas]


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------
def detectar_lineas(
    binaria: np.ndarray,
    quitar_renglones: bool = False,
    relleno_vertical: float = 0.30,
    relleno_horizontal: float = 0.01,
) -> list[Linea]:
    """Detecta las lineas de texto en la imagen binaria del folio (tinta = 255).

    quitar_renglones  : True si el papel es rayado/cuadriculado.
    relleno_vertical  : margen anadido arriba y abajo de cada linea, como fraccion de su alto
                        (para no cortar ascendentes ni descendentes al recortar).
    relleno_horizontal: margen anadido a izquierda y derecha, como fraccion del ancho del folio.
    """
    alto, ancho = binaria.shape[:2]
    tinta = _limpiar_ruido(binaria)
    if quitar_renglones:
        tinta = _limpiar_ruido(_quitar_renglones(tinta))

    # Unir palabras de una misma linea
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.04 * ancho)), 1))
    unida = cv2.dilate(tinta, k)

    # Perfil de proyeccion horizontal suavizado
    perfil = (unida > 0).sum(axis=1).astype(np.float32)
    ventana = max(3, (alto // 300) | 1)
    perfil = np.convolve(perfil, np.ones(ventana, dtype=np.float32) / ventana, mode="same")
    if perfil.max() == 0:
        return []

    umbral = max(3.0, 0.04 * float(perfil.max()))
    bandas = _bandas_de_filas(perfil, umbral, hueco_min=max(3, int(0.004 * alto)))

    lineas: list[Linea] = []
    alto_min = max(8, int(0.006 * alto))
    for y0, y1 in bandas:
        if y1 - y0 < alto_min:
            continue
        cols = np.where(tinta[y0:y1].any(axis=0))[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1]) + 1

        py = int(relleno_vertical * (y1 - y0))
        px = int(relleno_horizontal * ancho)
        x0, x1 = max(0, x0 - px), min(ancho, x1 + px)
        y0, y1 = max(0, y0 - py), min(alto, y1 + py)

        lineas.append(
            Linea(
                id=f"linea_{len(lineas) + 1:03d}",
                bbox=(x0, y0, x1, y1),
                bbox_norm=(x0 / ancho, y0 / alto, x1 / ancho, y1 / alto),
            )
        )

    if len(lineas) >= 3:
        altos = np.array([l.bbox[3] - l.bbox[1] for l in lineas])
        if (altos > 2.2 * np.median(altos)).any():
            warnings.warn(
                "Hay bandas mucho mas altas que el resto: puede que dos lineas se hayan fusionado "
                "(letra muy grande, descendentes tocando la linea siguiente...). Revisa con dibujar_lineas()."
            )
    return lineas


def dibujar_lineas(imagen: np.ndarray, lineas: list[Linea]) -> np.ndarray:
    """Copia de la imagen con las cajas dibujadas y numeradas, para comprobar visualmente."""
    out = imagen.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for i, l in enumerate(lineas, 1):
        x0, y0, x1, y1 = l.bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, str(i), (max(0, x0 - 40), y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return out


def a_segmentacion_kraken(lineas: list[Linea], nombre_imagen: str = "folio.png"):
    """Convierte las lineas al contenedor de segmentacion de Kraken (tipo 'bbox').

    Los campos siguen la documentacion de kraken.containers (Segmentation / BBoxLine).
    Si tu version de Kraken (>= 7.1) pide algun argumento distinto, el error saldra aqui.
    """
    from kraken.containers import BBoxLine, Segmentation  # import tardio: kraken es opcional

    return Segmentation(
        type="bbox",
        imagename=nombre_imagen,
        text_direction="horizontal-lr",
        script_detection=False,  
        lines=[BBoxLine(id=l.id, bbox=list(l.bbox), text_direction="horizontal-lr") for l in lineas],
        regions={},
        line_orders=[],
    )


if __name__ == "__main__":
    import sys

    from preprocesado import preprocesar_folio

    if len(sys.argv) < 2:
        sys.exit("Uso: python lineas.py foto.jpg [--rayado]")
    folio = preprocesar_folio(sys.argv[1])
    lineas = detectar_lineas(folio.binaria, quitar_renglones="--rayado" in sys.argv)
    print(f"Folio detectado: {folio.folio_detectado} | lineas encontradas: {len(lineas)}")
    for l in lineas:
        print(l.id, l.bbox, tuple(round(v, 3) for v in l.bbox_norm))
    cv2.imwrite("salida_lineas.png", dibujar_lineas(folio.color, lineas))
    print("Guardado salida_lineas.png")
