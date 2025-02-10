import argparse
import os
from PIL import Image

def load_act_palette(act_path):
    """
    Charge une palette depuis un fichier .act.
    On suppose que le fichier contient au moins 768 octets (256 couleurs x 3 octets).
    """
    with open(act_path, 'rb') as f:
        data = f.read()
    if len(data) < 768:
        raise ValueError("Fichier .act invalide : moins de 768 octets.")
    # On ne prend que les 768 premiers octets (R, G, B pour chaque couleur)
    palette = list(data[:768])
    return palette

def create_palette_image(palette):
    """
    Crée une image en mode 'P' et y charge la palette personnalisée.
    Cette image sera utilisée pour la quantification.
    """
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(palette)
    return pal_img

def quantize_and_save(image, out_name, pal_img):
    """
    Applique la quantification avec dithering Floyd–Steinberg
    en utilisant la palette contenue dans pal_img, convertit en RGB (24-bit)
    et sauvegarde l'image au format BMP.
    """
    quantized = image.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
    final_img = quantized.convert("RGB")
    final_img.save(out_name, format="BMP")
    print(f"Image sauvegardée : {out_name}")

def process_image(input_path, output_base, target_width=800, target_height=480):
    """
    Traite l'image selon la logique suivante :
    
    1. Redimensionnement proportionnel pour que la largeur soit 800 pixels.
    2. Si la hauteur obtenue (new_height) est >= 480 :
         - Génère des crops de 800x480 en glissant verticalement la fenêtre de crop.
         - En plus, crée une image forcée (non proportionnelle) en redimensionnant
           l'image initiale (après redimensionnement proportionnel) en 800x480.
       Sinon :
         - Redimensionnement forcé en 800x480 (déformation possible).
    3. Chaque image est quantifiée via la palette "N-color.act" avec dithering
       et sauvegardée en BMP 24-bit.
    """
    # Ouvrir l'image d'entrée et la convertir en RGB
    img = Image.open(input_path).convert("RGB")
    orig_width, orig_height = img.size

    # Redimensionnement proportionnel pour obtenir une largeur de 800 pixels
    new_width = target_width
    new_height = int(orig_height * (target_width / orig_width))
    
    # Charger la palette
    palette = load_act_palette("N-color.act")
    pal_img = create_palette_image(palette)
    
    if new_height >= target_height:
        # Redimensionnement proportionnel
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"Image redimensionnée proportionnellement : {new_width}x{new_height}")
        
        # Génération des crops par glissement vertical
        y_max = new_height - target_height
        # On calcule 4 positions : en haut, 1/3, 2/3 et en bas (on élimine les doublons)
        crop_positions = sorted(set([0, round(y_max/3), round(2*y_max/3), y_max]))
        
        # Suffixes pour nommer les différentes découpes
        suffixes = ["top", "upper", "lower", "bottom"]
        num_crops = len(crop_positions)
        
        for i, y in enumerate(crop_positions):
            crop_box = (0, y, target_width, y + target_height)
            crop_img = img_resized.crop(crop_box)
            if num_crops == 1:
                out_name = f"{output_base}.bmp"
            else:
                suffix = suffixes[i] if i < len(suffixes) else str(i)
                out_name = f"{output_base}_{suffix}.bmp"
            quantize_and_save(crop_img, out_name, pal_img)
        
        # En plus, créer une image forcée redimensionnée en 800x480
        forced_img = img_resized.resize((target_width, target_height), Image.LANCZOS)
        out_name = f"{output_base}_resized.bmp"
        quantize_and_save(forced_img, out_name, pal_img)
    
    else:
        # Si new_height < 480, faire un redimensionnement forcé pour obtenir 800x480
        forced_img = img.resize((target_width, target_height), Image.LANCZOS)
        print(f"Hauteur insuffisante ({new_height} < {target_height}), redimensionnement forcé en {target_width}x{target_height}")
        out_name = f"{output_base}.bmp"
        quantize_and_save(forced_img, out_name, pal_img)

def main():
    parser = argparse.ArgumentParser(
        description=("Redimensionne l'image pour une largeur de 800 pixels, "
                     "puis :\n- si la hauteur obtenue est >=480, génère plusieurs crops "
                     "de 800x480 par glissement vertical et en plus une image forcée redimensionnée en 800x480 ;\n"
                     "- sinon, effectue un redimensionnement forcé en 800x480.\n"
                     "La quantification se fait avec dithering Floyd–Steinberg via la palette 'N-color.act'. "
                     "La sortie est toujours un BMP 24-bit.")
    )
    parser.add_argument("input_image",
                        help="Chemin vers l'image d'entrée (ex: image.png)")
    parser.add_argument("output_image", nargs="?", default=None,
                        help=("Nom de base pour l'image de sortie. "
                              "Si non spécifié, le nom de base est dérivé du nom de l'image d'entrée."))
    
    args = parser.parse_args()
    
    if args.output_image is None:
        output_base = os.path.splitext(args.input_image)[0]
    else:
        output_base = os.path.splitext(args.output_image)[0]
    
    process_image(args.input_image, output_base, target_width=800, target_height=480)

if __name__ == "__main__":
    main()