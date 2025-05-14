import numpy as np
from scipy.spatial.distance import cosine, euclidean
import cv2
import mediapipe as mp
import os

# MediaPipe arcjellemzők kinyeréséhez
mp_face_mesh = mp.solutions.face_mesh

# Szigorúbb küszöbérték (kisebb érték = szigorúbb összehasonlítás)
# Ezt az értéket szükség szerint tesztelés alapján módosíthatod
THRESHOLD = 0.018  # Csökkentve 0.15-ről

# Több, megkülönböztető arcjellemző pont definiálása
# Ezek több pontot tartalmaznak a szemek, orr, állvonal és száj körül
ARCJELLEMZO_PONTOK = [
    # Szemek (részletesebben)
    33, 133, 160, 158, 153, 144, 362, 385, 387, 380, 373, 263, 466,
    # Szemöldökök
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Orrhát és orrhegy
    6, 197, 195, 5, 4, 1, 19, 94,
    # Szájsarkok és ajkak
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Állvonal (több pont)
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


def extract_facial_features(image):
    """
    Arcjellemzők kinyerése egy képről javított jellemzőpont-kiválasztással MediaPipe Face Mesh használatával.

    Paraméterek:
    - image: A feldolgozandó kép (numpy tömb, CV2 formátum)

    Visszatérési érték:
    - Normalizált arcjellemzők vektora vagy None, ha nem található arc
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6) as face_mesh:  # Megnövelt felismerési küszöbérték

        # RGB konverzió (MediaPipe RGB-t vár)
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Arcjellemzők kinyerése
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        # A kiválasztott arcjellemző pontok kinyerése
        landmarks = []
        face_landmarks = results.multi_face_landmarks[0]

        for idx in ARCJELLEMZO_PONTOK:
            landmark = face_landmarks.landmark[idx]
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Konvertálás numpy tömbbé
        landmarks_array = np.array(landmarks)

        # Középpont kiszámítása
        center = np.mean(landmarks_array, axis=0)

        # Normalizálás a középponthoz viszonyítva
        normalized_landmarks = landmarks_array - center

        # Méret szerinti normalizálás
        max_distance = np.max(np.linalg.norm(normalized_landmarks, axis=1))
        if max_distance > 0:
            normalized_landmarks = normalized_landmarks / max_distance

        # Másodrendű jellemzők hozzáadása (távolságok kulcspontok között)
        # Ez segít az arcvonások közötti kapcsolati információk rögzítésében
        features = normalized_landmarks.flatten()

        # Távolságok számítása kulcsfontosságú arcpontok között
        # Az ARCJELLEMZO_PONTOK indexeit használva, amelyek megfelelnek:
        # Szemsarkoknak, orrhegynek, szájsarkoknak
        szem_indexek = [0,
                        12]  # Feltételezve, hogy ezek az indexek a bal és jobb szemnek felelnek meg a kiválasztott pontjainkban
        orr_indexek = [26]  # Feltételezve, hogy ez az index az orrhegynek felel meg
        szaj_indexek = [30, 33]  # Feltételezve, hogy ezek az indexek a szájsarkoknak felelnek meg

        # Távolságok számítása szemek, orr és száj között
        for i in szem_indexek:
            for j in orr_indexek:
                dist = np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j])
                features = np.append(features, dist)

        for i in szem_indexek:
            for j in szaj_indexek:
                dist = np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j])
                features = np.append(features, dist)

        for i in orr_indexek:
            for j in szaj_indexek:
                dist = np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j])
                features = np.append(features, dist)

        return features


def compare_faces(features1, features2, threshold=THRESHOLD):
    """
    Két arcjellemző vektor összehasonlítása több távolságmérték használatával.

    Paraméterek:
    - features1: Az első arc jellemzői (numpy tömb)
    - features2: A második arc jellemzői (numpy tömb)
    - threshold: Hasonlósági küszöbérték (0-1 között, kisebb = szigorúbb)

    Visszatérési érték:
    - (is_same, distance): Tuple, ahol
      - is_same: Boolean, True ha a két arc ugyanahhoz a személyhez tartozik
      - distance: Kombinált távolságérték (kisebb = hasonlóbb)
    """
    if features1 is None or features2 is None:
        return False, 1.0

    # Jellemzők dimenzióinak ellenőrzése
    if len(features1) != len(features2):
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]

    try:
        # Koszinusz és euklideszi távolságok számítása (normalizáljuk az euklideszi távolságot)
        cosine_dist = cosine(features1, features2)

        # Euklideszi távolság normalizálása a jellemzők hosszával osztva
        euclidean_dist = euclidean(features1, features2) / len(features1)

        # Távolságok kombinálása (súlyozott átlag)
        # Nagyobb súlyt adunk a koszinusz távolságnak az általános forma miatt,
        # de figyelembe vesszük az euklideszi távolságot is az abszolút különbségek miatt
        combined_distance = 0.7 * cosine_dist + 0.3 * euclidean_dist

        # Összehasonlítás a küszöbértékkel
        is_same = combined_distance < threshold

        return is_same, combined_distance
    except Exception as e:
        print(f"Hiba a távolság számításakor: {e}")
        return False, 1.0


def compare_face_images(image1_path, image2_path, threshold=THRESHOLD):
    """
    Két kép összehasonlítása arcfelismerés alapján.

    Paraméterek:
    - image1_path: Az első kép elérési útja
    - image2_path: A második kép elérési útja
    - threshold: Hasonlósági küszöbérték

    Visszatérési érték:
    - (is_same, distance): Tuple, ahol
      - is_same: Boolean, True ha a két arc ugyanahhoz a személyhez tartozik
      - distance: Távolság értéke (kisebb = hasonlóbb)
    """
    try:
        # Képek betöltése
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            print("Hiba: Nem sikerült betölteni legalább az egyik képet.")
            return False, 1.0

        # Arcjellemzők kinyerése
        features1 = extract_facial_features(image1)
        features2 = extract_facial_features(image2)

        if features1 is None or features2 is None:
            print("Hiba: Nem sikerült kinyerni az arcjellemzőket legalább az egyik képről.")
            return False, 1.0

        # Arcok összehasonlítása
        return compare_faces(features1, features2, threshold)

    except Exception as e:
        print(f"Hiba a képek összehasonlításakor: {e}")
        return False, 1.0


def compare_multiple_faces(image_paths, threshold=THRESHOLD):
    """
    Több kép összehasonlítása arcfelismerés alapján.

    Paraméterek:
    - image_paths: Képek elérési útjainak listája
    - threshold: Hasonlósági küszöbérték

    Visszatérési érték:
    - Dictionary a következő kulcsokkal:
      - 'comparison_results': Lista, amely az összehasonlítási eredményeket tartalmazza
      - 'same_person_count': Hány összehasonlításban találtunk azonos személyt
      - 'total_comparisons': Az összes elvégzett összehasonlítás száma
    """
    # Ellenőrizzük a képek számát
    if len(image_paths) < 2:
        print("Hiba: Legalább 2 kép szükséges az összehasonlításhoz.")
        return {
            'comparison_results': [],
            'same_person_count': 0,
            'total_comparisons': 0
        }

    # Arcjellemzők kinyerése minden képről
    features_list = []
    for img_path in image_paths:
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Hiba: Nem sikerült betölteni a képet: {img_path}")
                features_list.append(None)
                continue

            features = extract_facial_features(image)
            features_list.append(features)

            if features is None:
                print(f"Figyelmeztetés: Nem található arc a képen: {img_path}")

        except Exception as e:
            print(f"Hiba a kép feldolgozásakor: {img_path}, {e}")
            features_list.append(None)

    # Összehasonlítási eredmények gyűjteménye
    comparison_results = []
    same_person_count = 0
    total_comparisons = 0

    # Minden képpár összehasonlítása
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            if features_list[i] is None or features_list[j] is None:
                continue

            is_same, distance = compare_faces(features_list[i], features_list[j], threshold)
            total_comparisons += 1

            result = {
                'image1_index': i,
                'image2_index': j,
                'image1_path': image_paths[i],
                'image2_path': image_paths[j],
                'is_same_person': is_same,
                'distance': distance
            }

            comparison_results.append(result)

            if is_same:
                same_person_count += 1

    return {
        'comparison_results': comparison_results,
        'same_person_count': same_person_count,
        'total_comparisons': total_comparisons
    }


# Példa használat:
if __name__ == "__main__":
    # Több kép összehasonlítása
    images = ["imgt1.png", "imgt2.png", "img_2.png"]
    results = compare_multiple_faces(images)

    print("\n----- Összehasonlítási eredmények -----")
    for result in results['comparison_results']:
        print(f"Kép {result['image1_index'] + 1} és Kép {result['image2_index'] + 1}: "
              f"{'Azonos személy' if result['is_same_person'] else 'Különböző személyek'} "
              f"(távolság: {result['distance']:.4f})")

    print("\n----- Végső eredmény -----")

    if results['total_comparisons'] == 0:
        print("Nem sikerült elvégezni az összehasonlítást.")
    elif results['same_person_count'] == results['total_comparisons']:
        eredmeny = True
        print(eredmeny)
    elif results['same_person_count'] == 0:
        eredmeny = False
        print(eredmeny)
    else:
        print(f"Vegyes eredmény: {results['same_person_count']}/{results['total_comparisons']} esetben azonos személy.")
        eredmeny = False
        print(eredmeny)



def visualize_face_landmarks(image_path, output_path=None):
    """
    Arcjellemző pontok vizualizálása egy képen a hibakeresés segítéséhez.

    Paraméterek:
    - image_path: A bemeneti kép elérési útja
    - output_path: A kimeneti kép mentési útja (ha None, akkor megjeleníti a képet)

    Visszatérési érték:
    - None (megjeleníti vagy menti a képet)
    """
    try:
        # Kép betöltése
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hiba: Nem sikerült betölteni a képet: {image_path}")
            return

        # RGB konverzió a vizualizációhoz
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dimenziók lekérése
        h, w, _ = image.shape

        # Másolat készítése a rajzoláshoz
        vis_image = image.copy()

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6) as face_mesh:

            # Kép feldolgozása
            results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                print(f"A(z) {image_path} képen nem található arc")
                return

            # Jellemző pontok rajzolása
            face_landmarks = results.multi_face_landmarks[0]

            # Kiválasztott jellemző pontok rajzolása különböző színekkel
            for idx in ARCJELLEMZO_PONTOK:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Zöld a kiválasztott jellemző pontokhoz

            # Néhány kulcsjellemző pont címkéjének hozzáadása
            for i, idx in enumerate(ARCJELLEMZO_PONTOK[:5]):  # Csak néhány kulcspont címkézése
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(vis_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Kép mentése vagy megjelenítése
            if output_path:
                cv2.imwrite(output_path, vis_image)
                print(f"Vizualizáció mentve ide: {output_path}")
            else:
                cv2.imshow("Arcjellemző pontok", vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        print(f"Hiba a jellemző pontok vizualizációja során: {e}")