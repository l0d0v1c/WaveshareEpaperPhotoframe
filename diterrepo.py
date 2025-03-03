import os
from PIL import Image

def load_act_palette(act_path):
    """
    Charge une palette depuis un fichier .act.
    """
    with open(act_path, 'rb') as f:
        data = f.read()
    if len(data) < 768:
        raise ValueError("Fichier .act invalide : moins de 768 octets.")
    palette = list(data[:768])
    return palette

def create_palette_image(palette):
    """
    Crée une image en mode 'P' et y charge la palette personnalisée.
    """
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(palette)
    return pal_img

def quantize_and_save(image, out_name, pal_img):
    """
    Applique la quantification et sauvegarde en BMP.
    """
    quantized = image.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
    final_img = quantized.convert("RGB")
    final_img.save(out_name, format="BMP")
    print(f"Image sauvegardée : {out_name}")

def process_images_in_directory(directory, target_width=800, target_height=480):
    """
    Traite toutes les images JPG/JPEG dans un répertoire.
    """
    palette = load_act_palette("N-color.act")
    pal_img = create_palette_image(palette)
    
    images = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg"))]
    
    for i, image_name in enumerate(images, start=1):
        input_path = os.path.join(directory, image_name)
        output_name = os.path.join(directory, f"{i}.bmp")
        
        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
        
        quantize_and_save(img_resized, output_name, pal_img)

if __name__ == "__main__":
    repo_directory = "repo"  # Nom du répertoire contenant les images
    if os.path.exists(repo_directory) and os.path.isdir(repo_directory):
        process_images_in_directory(repo_directory)
    else:
        print(f"Le répertoire '{repo_directory}' n'existe pas.")
