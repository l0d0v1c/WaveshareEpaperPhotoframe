#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def load_palette(act_file):
    """
    Charge la palette de couleurs à partir d'un fichier ACT.
    Dans le cas d'un fichier de 772 octets (palette Adobe à 256 couleurs),
    on ignore les 4 derniers octets.
    
    :param act_file: Chemin vers le fichier .act
    :return: Un tableau numpy de forme (n, 3) contenant les couleurs.
    """
    with open(act_file, "rb") as f:
        data = f.read()
    
    # Si le fichier fait 772 octets, on considère que seuls les 768 premiers sont utiles.
    if len(data) == 772:
        data = data[:768]
    
    if len(data) % 3 != 0:
        raise ValueError("La taille du fichier ACT n'est pas un multiple de 3 (taille : {}).".format(len(data)))
    
    num_colors = len(data) // 3
    palette = np.frombuffer(data, dtype=np.uint8).reshape((num_colors, 3))
    return palette

def apply_palette(image, palette):
    """
    Remappe chaque pixel de l'image sur la couleur de la palette la plus proche.

    :param image: Image d'entrée (tableau numpy en BGR)
    :param palette: Tableau numpy de forme (n, 3) contenant la palette de couleurs.
    :return: Image quantifiée aux couleurs de la palette.
    """
    h, w, _ = image.shape
    image_flat = image.reshape(-1, 3).astype(np.int16)
    palette = palette.astype(np.int16)
    
    # Calcul vectorisé des distances (distance euclidienne au carré)
    diff = image_flat[:, np.newaxis, :] - palette[np.newaxis, :, :]  # forme : (N_pixels, n_couleurs, 3)
    distances = np.sum(diff ** 2, axis=2)  # forme : (N_pixels, n_couleurs)
    
    indices = np.argmin(distances, axis=1)
    quantized_flat = palette[indices]
    
    quantized_image = quantized_flat.reshape(h, w, 3).astype(np.uint8)
    return quantized_image

def detect_edges(image):
    """
    Détecte les contours sur l'image en niveaux de gris grâce à l'algorithme de Canny.
    
    :param image: Image d'entrée.
    :return: Image binaire contenant les contours.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def overlay_edges(image, edges):
    """
    Superpose les contours détectés sur l'image en remplaçant les pixels correspondants par du noir.
    
    :param image: Image en couleur.
    :param edges: Image binaire (résultat de Canny).
    :return: Image finale avec les contours en noir.
    """
    result = image.copy()
    result[edges != 0] = [0, 0, 0]
    return result

def stained_glass_effect(input_path, act_path, output_path):
    """
    Applique l'effet vitrail à une image :
      - Charge l'image d'entrée.
      - Charge la palette de couleurs depuis le fichier ACT.
      - Remappe chaque pixel sur la couleur de la palette la plus proche.
      - Détecte les contours sur l'image quantifiée.
      - Superpose les contours (en noir) sur l'image quantifiée.
      - Sauvegarde l'image finale.

    :param input_path: Chemin vers l'image d'entrée.
    :param act_path: Chemin vers le fichier de palette (N_color.act).
    :param output_path: Chemin pour sauvegarder l'image de sortie.
    """
    # Chargement de l'image
    img = cv2.imread(input_path)
    if img is None:
        print("Erreur : Impossible de charger l'image", input_path)
        return

    # Chargement de la palette
    palette = load_palette(act_path)
    
    if palette.shape[0] < 8:
        print("Erreur : La palette doit contenir au moins 8 couleurs.")
        return
    elif palette.shape[0] > 8:
        print("Attention : La palette contient {} couleurs, on utilisera les 8 premières.".format(palette.shape[0]))
        palette = palette[:8]

    # Remappage de l'image sur la palette (quantification)
    quantized_img = apply_palette(img, palette)

    # Détection des contours sur l'image quantifiée
    edges = detect_edges(quantized_img)

    # Superposition des contours pour l'effet vitrail
    result_img = overlay_edges(quantized_img, edges)

    # Sauvegarde de l'image finale
    cv2.imwrite(output_path, result_img)
    print("Image sauvegardée sous", output_path)

if __name__ == '__main__':
    # Spécifiez ici le chemin de votre image d'entrée,
    # le fichier de palette (N_color.act) et le fichier de sortie.
    input_image = "test3.jpg"            # Remplacez par le chemin de votre image
    act_file = "N-color.act"               # Chemin vers le fichier de palette
    output_image = "test3_vitrail.jpg"    # Nom du fichier de sortie

    stained_glass_effect(input_image, act_file, output_image)