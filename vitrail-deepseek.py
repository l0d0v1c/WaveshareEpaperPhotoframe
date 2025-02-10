import cv2
import numpy as np

def load_color_palette(act_file):

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

def image_to_custom_palette(input_image_path, output_folder=".", palette_path="N-color.act"):
    # Charger l'image
    img = cv2.imread(input_image_path)
    if img is None:
        print("Impossible de charger l'image.")
        return
    
    # Charger la palette personnalisée
    custom_palette = load_color_palette(palette_path)
    
    # Conversion en format LAB pour une meilleure perception des couleurs
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Redimensionner l'image (optionnel) pour accélérer le traitement
    height, width = img.shape[:2]
    img_resized = cv2.resize(img_lab, (width // 4, height // 4))
    
    # Transformer en tableau numpy
    data = np.reshape(img_resized, (-1, 3))
    
    # Convertir la palette personnalisée en LAB pour cohérence
    custom_palette_lab = []
    for color in custom_palette:
        # Conversion de RGB à LAB
        rgb_color = np.array([[color]], dtype=np.uint8)
        lab_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2LAB)
        custom_palette_lab.append(lab_color[0][0])
    custom_palette_lab = np.array(custom_palette_lab)
    
    # Calculer la distance Euclidienne entre chaque pixel et les couleurs de la palette
    distances = np.zeros((len(data), len(custom_palette_lab)))
    for i, color in enumerate(custom_palette_lab):
        distances[:, i] = np.sum((data - color) ** 2, axis=1)
    
    # Trouver l'index de la couleur de palette la plus proche pour chaque pixel
    labels = np.argmin(distances, axis=1)
    
    # Remplacer chaque pixel par la couleur de palette correspondante (en LAB)
    segmented_data = custom_palette_lab[labels]
    
    # Redimensionner l'image segmentée pour qu'elle corresponde à l'original
    segmented_img = np.reshape(segmented_data, (height // 4, width // 4, 3))
    segmented_img = cv2.resize(segmented_img, (width, height))
    
    # Convertir de LAB à BGR
    segmented_img_bgr = cv2.cvtColor(segmented_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Générer le nom du fichier de sortie
    input_filename = input_image_path.split('/')[-1]
    output_filename = f"DSvitrail_{input_filename}"
    output_path = f"{output_folder}/{output_filename}"
    
    # Sauvegarder l'image résultat
    cv2.imwrite(output_path, segmented_img_bgr)

# Utiliser le script avec votre image et votre palette
input_path = "test3.jpg"       # Remplacez par le chemin de votre image
image_to_custom_palette(input_path)
