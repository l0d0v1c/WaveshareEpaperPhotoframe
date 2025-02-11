import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

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

def apply_stained_glass_effect(image_path, palette_file):
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Charger la palette de couleurs
    palette = load_palette(palette_file)
    num_colors = len(palette)

    # Réduire les couleurs avec KMeans en utilisant la palette
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Assigner les couleurs de la palette
    labels = kmeans.labels_
    reduced_color_image = palette[labels].reshape(image.shape).astype(np.uint8)

    # Convertir l'image en espace couleur BGR
    reduced_color_image = cv2.cvtColor(reduced_color_image, cv2.COLOR_LAB2BGR)

    # Appliquer un flou gaussien pour lisser les régions
    blurred_image = gaussian_filter(reduced_color_image, sigma=3)

    # Convertir en niveaux de gris pour détecter les contours
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Détecter les contours
    edges = cv2.Canny(gray_image, 100, 200)

    # Dilater les contours pour un effet vitrail plus prononcé
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Inverser les contours pour les rendre noirs
    contours = cv2.bitwise_not(edges)

    # Appliquer les contours sur l'image réduite en couleurs
    stained_glass_image = cv2.bitwise_and(blurred_image, blurred_image, mask=contours)

    # Sauvegarder l'image résultante
    output_path = "stained_glass_effect.png"
    cv2.imwrite(output_path, stained_glass_image)
    print(f"Image sauvegardée sous {output_path}")

# Exemple d'utilisation
apply_stained_glass_effect("test3.jpg", palette_file="N-color.act")
