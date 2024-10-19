import time
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
import requests
from bs4 import BeautifulSoup
from moviepy.editor import ImageSequenceClip, AudioFileClip
from PIL import Image
import imagehash
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from moviepy.editor import ImageSequenceClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, concatenate_audioclips
import cv2
import numpy as np
import tensorflow as tf

def enhance_image_esrgan(image_path, model_dir="esrgan-tf2-tensorflow2-esrgan-tf2-v1/", output_folder="enhanced_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        model = tf.saved_model.load(model_dir)

        img = Image.open(image_path).convert('RGB')
        
        img = img.resize((384, 384))  # Taille d'entrée du modèle à ajuster si besoin
        
        img = np.array(img) / 255.0
        
        img = np.expand_dims(img, axis=0)
        
        img = img.astype(np.float32)
        
        img_enhanced = model(img, training=False)

        img_enhanced = img_enhanced[0].numpy()
        img_enhanced = np.clip(img_enhanced, 0, 1)
        img_enhanced = (img_enhanced * 255).astype(np.uint8)

        if img_enhanced.shape[-1] == 3:
            img_enhanced = img_enhanced[..., :3]  # On s'assure d'avoir seulement les 3 canaux

        img_enhanced = Image.fromarray(img_enhanced, 'RGB')

        enhanced_image_path = os.path.join(output_folder, os.path.basename(image_path))
        img_enhanced.save(enhanced_image_path, format='PNG')

        print(f"Image améliorée sauvegardée : {enhanced_image_path}")
        return enhanced_image_path

    except Exception as e:
        print(f"Erreur lors de l'amélioration de l'image {image_path}: {e}")
        return None

# def enhance_image_esrgan(image_path, model_dir = "esrgan-tf2-tensorflow2-esrgan-tf2-v1/", output_folder="enhanced_images"):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     try:
#         # Charger le modèle ESRGAN depuis le répertoire contenant saved_model.pb
#         model = tf.saved_model.load(model_dir)

#         # Charger l'image
#         img = Image.open(image_path)
#         img = img.resize((384, 384))  # Redimensionner à la taille d'entrée du modèle, modifie cette valeur si nécessaire
#         img = np.array(img) / 127.5 - 1  # Normaliser l'image entre -1 et 1 (format souvent utilisé pour les réseaux de neurones)

#         # Ajouter une dimension (batch size de 1) pour correspondre aux exigences du modèle
#         img = np.expand_dims(img, axis=0)

#         # Appliquer la super-résolution avec le modèle
#         img_enhanced = model(img)

#         # Post-traitement de l'image pour revenir à un format exploitable (0-255)
#         img_enhanced = ((img_enhanced[0].numpy() + 1) * 127.5).astype(np.uint8)

#         # Convertir l'image traitée en objet PIL
#         img_enhanced = Image.fromarray(img_enhanced)

#         # Sauvegarder l'image améliorée
#         enhanced_image_path = os.path.join(output_folder, os.path.basename(image_path))
#         img_enhanced.save(enhanced_image_path, format='PNG')

#         print(f"Image améliorée sauvegardée : {enhanced_image_path}")
#         return enhanced_image_path
#     except Exception as e:
#         print(f"Erreur lors de l'amélioration de l'image {image_path}: {e}")
#         return None


# from realesrgan import RealESRGANer


# def enhance_image_realesrgan(image_path, output_folder="enhanced_images"):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     try:
#         # Charger l'image basse résolution
#         img = Image.open(image_path)

#         # Initialiser le modèle Real-ESRGAN avec GPU si disponible (sinon CPU)
#         model = RealESRGANer.from_pretrained('RealESRGAN_x2plus') # or x4
        
#         # Appliquer la super-résolution à l'image
#         img_enhanced = model.predict(img)

#         # Sauvegarder l'image améliorée
#         enhanced_image_path = os.path.join(output_folder, os.path.basename(image_path))
#         img_enhanced.save(enhanced_image_path, format='PNG')
        
#         print(f"Image améliorée sauvegardée : {enhanced_image_path}")
#         return enhanced_image_path
#     except Exception as e:
#         print(f"Erreur lors de l'amélioration de l'image {image_path}: {e}")
#         return None


def enhance_image_pillow(image_path, upscale_factor=2, output_folder="enhanced_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        img = Image.open(image_path)

        new_width = int(img.width * upscale_factor)
        new_height = int(img.height * upscale_factor)
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)  # Utilisation du filtre LANCZOS pour un redimensionnement de haute qualité

        img_sharpened = img_resized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))  # Appliquer un filtre de netteté

        enhancer = ImageEnhance.Contrast(img_sharpened)
        img_contrast = enhancer.enhance(1.3)  # Augmenter légèrement le contraste pour rendre l'image plus vive

        enhanced_image_path = os.path.join(output_folder, os.path.basename(image_path))
        img_contrast.save(enhanced_image_path, format='JPEG', quality=95)  # Sauvegarder avec une qualité de compression élevée

        return enhanced_image_path
    except Exception as e:
        print(f"Erreur lors de l'amélioration de l'image {image_path}: {e}")
        return None


def enhance_image_cv2(image_path, upscale_factor=2, output_folder="enhanced_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Charger l'image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erreur : impossible de charger l'image {image_path}")
        return None
    
    # 1. Redimensionnement avec l'interpolation bicubique
    new_width = int(image.shape[1] * upscale_factor)
    new_height = int(image.shape[0] * upscale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 2. Appliquer un filtre de netteté
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Filtre de netteté
    sharpened_image = cv2.filter2D(src=resized_image, ddepth=-1, kernel=kernel)

    # 3. Réduction du bruit
    denoised_image = cv2.fastNlMeansDenoisingColored(sharpened_image, None, 10, 10, 7, 21)

    # Sauvegarder l'image améliorée
    enhanced_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(enhanced_image_path, denoised_image)
    
    return enhanced_image_path


# Fonction pour générer des combinaisons de mots avec OpenAI (API 1.x.x)
def generate_combinations(base_word):
    # response = client.chat.completions.create(model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": f"Generate different phrases and combinations using the word '{base_word}'"}
    # ])

    # combinations_text = response.choices[0].message.content
    # combinations = combinations_text.strip().split("\n")
 
    return [base_word]
    # return [base_word] + combinations[:5]

def search_google_images(query, n_images=1000, images_per_page=100, safe_search_activated = False):
    image_urls = []
    start = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    safe_search_param = "off" if not safe_search_activated else "active"
    n_images += 20 # for potential errors
    
    while len(image_urls) < n_images:
        url = f"https://www.google.com/search?tbm=isch&q={query}&start={start}&safe={safe_search_param}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Erreur lors de la requête: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all("img")
        
        for img_tag in image_tags:
            if len(image_urls) >= n_images:
                break
            try:
                img_url = img_tag['src']
                image_urls.append(img_url)
            except KeyError:
                continue

        print(len(image_urls))
        start += images_per_page 
        time.sleep(2)   


    print(len(image_urls))
    return image_urls 

def download_images(image_urls, download_folder="images"):
    if os.path.exists(download_folder):
        shutil.rmtree(download_folder) 
        
    os.makedirs(download_folder)
    
    local_files = []
    for i, url in enumerate(image_urls):
        try:
            img_data = requests.get(url).content
            file_name = os.path.join(download_folder, f"image_{i}.jpg")
            with open(file_name, 'wb') as handler:
                handler.write(img_data)
            local_files.append(file_name)
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image : {e}")
    
    return local_files

def remove_duplicate_images(image_files, n_images = 1000):
    unique_images = []
    hashes = set()
    print(len(image_files))

    # not working well tbh
    for image_file in image_files:
        if len(unique_images) < n_images:
            img = Image.open(image_file)
            img_hash = imagehash.phash(img)
            if img_hash not in hashes:
                hashes.add(img_hash)
                unique_images.append(image_file)

    print(len(unique_images))

    return unique_images

def loop_audio(audio_clips, duration):
    if not isinstance(audio_clips, list):
        audio_clips = [audio_clips]

    concatenated_clip = concatenate_audioclips(audio_clips)
    
    clips = []
    total_duration = 0
    while total_duration < duration:
        clips.append(concatenated_clip)
        total_duration += concatenated_clip.duration
    
    return concatenate_audioclips(clips).subclip(0, duration)

def resize_image(image_path, size=(800, 600)):
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_resized = img.resize(size, Image.ANTIALIAS)
    img_resized.save(image_path, format='JPEG')
    
    return image_path

def generate_text_image(text, size=(800, 600), font_size=60):
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            print("Aucune police trouvée, utilisation de la police par défaut qui ne supporte pas le réglage de taille")
            font = ImageFont.load_default()
    
    text_width, text_height = draw.textsize(text, font=font)
    # position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    position = ((size[0] - text_width) // 2, size[1] - text_height - 10)

    draw.text(position, text, font=font, fill=(0, 0, 0, 255))
    
    image_path = f"/tmp/{text}.png"
    img.save(image_path, "PNG")
    
    return image_path


def create_video(image_files, audio_file,keyword, duration_per_image=2):
    output_file="videos/"+keyword+".mp4"
    resized_images = [resize_image(image) for image in image_files]
    
    clip = ImageSequenceClip(resized_images, fps=1/duration_per_image)
    
    audio = AudioFileClip(audio_file)
    looped_audio = loop_audio([audio], clip.duration)
    
    clips_with_text = []
    
    num_images = len(image_files)
    
    for i in range(num_images):
        top_text = f"TOP {num_images - i}"
        text_image_path = generate_text_image(top_text)
        text_clip = ImageSequenceClip([text_image_path], durations=[duration_per_image])
        img_clip = clip.subclip(i * duration_per_image, (i + 1) * duration_per_image)
        video_with_text = CompositeVideoClip([img_clip, text_clip.set_position('center')])
        clips_with_text.append(video_with_text)
    
    final_clip = concatenate_videoclips(clips_with_text)
    final_clip = final_clip.set_audio(looped_audio)
    final_clip.write_videofile(output_file, codec="libx264")

# Fonction principale
def create_shitpost(keyword, nb):
    print(f"Génération de combinaisons pour : {keyword}")
    combinations = generate_combinations(keyword)

    all_images = []
    for comb in combinations:
        print(f"Recherche d'images pour : {comb}")
        image_urls = search_google_images(comb, n_images= nb)
        all_images.extend(image_urls)

    print(f"Téléchargement des images...")
    image_files = download_images(all_images)

    print("Vérification des doublons...")
    unique_images = remove_duplicate_images(image_files, n_images= nb)
    
    print("Amélioration de la qualité des images...")
    enhanced_images = [enhance_image_cv2(img) for img in unique_images]
    # enhanced_images = [enhance_image_pillow(img) for img in unique_images]
    # enhanced_images = [enhance_image_realesrgan(img) for img in unique_images]
    # enhanced_images = [enhance_image_esrgan(img) for img in unique_images]

    audio_file = "musics/ncs1.mp3" 
    print("Création de la vidéo...")
    create_video(enhanced_images, audio_file, keyword)
    print("Vidéo créée avec succès !")


keyword = input("Entrez un mot clé pour le shitpost: ")
nb = int(input("Entrez le nombre d'images pour le shitpost: "))

create_shitpost(keyword, nb)
