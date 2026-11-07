# DeepECG — application Android

Application de démonstration qui photographie une impression ECG sur papier
millimétré, en extrait le tracé, puis le classe avec un modèle DeepECG exporté
en TensorFlow Lite.

Classes prédites (PhysioNet/CinC Challenge 2017) : **Normal**, **Fibrillation
atriale**, **Autre**, **Bruité**.

> ⚠️ Démo de recherche uniquement. Ce n'est **pas un dispositif médical** et
> le résultat ne remplace en aucun cas un avis médical.

## Principe

1. **Photo** de l'impression ECG (appareil photo ou galerie).
2. **Numérisation** (`EcgDigitizer`) : seuillage du tracé sombre, position du
   tracé colonne par colonne, calibration temps/amplitude grâce à la grille
   rose (25 mm/s, 300 Hz, 10 mm/mV), conversion en mV. Si la grille n'est pas
   détectée, la largeur de la photo est supposée représenter 30 s et
   l'amplitude est estimée (indiqué dans l'interface).
3. **Classification** (`EcgClassifier`) : le signal est rééchantillonné sur la
   longueur d'entrée du modèle (9000 points pour `CNN_ECG.py`, 10100 pour
   `Conv1D_ECG.py` avec normalisation z-score) puis passé au modèle TFLite
   embarqué, qui renvoie les 4 probabilités.

## Prérequis

- Android Studio (ou un SDK Android + JDK 17) ;
- un modèle entraîné avec ce dépôt (par ex. `Keras_models/my_model.h5` après
  `python CNN_ECG.py cinc`) — **aucun poids entraîné n'est fourni dans ce dépôt** ;
- Python 3 + TensorFlow 2.x pour l'export.

## Installation

1. Exportez le modèle vers TFLite, depuis la racine du dépôt :

   ```bash
   pip install tensorflow
   python export_tflite.py --model Keras_models/my_model.h5 --arch cnn
   # ou --arch conv1d pour un modèle entraîné avec Conv1D_ECG.py
   ```

   Cela écrit `android/app/src/main/assets/ecg_model.tflite` et
   `model_info.txt` (forme d'entrée + normalisation).

2. Ouvrez le dossier `android/` dans Android Studio, laissez Gradle
   synchroniser, puis lancez l'application sur un appareil ou un émulateur
   (minSdk 24).

3. Dans l'application : « Prendre une photo » (ou « Choisir une image »),
   cadrez toute la largeur du tracé ECG, bien à plat et sans reflet.

## Structure

```
android/
├── app/src/main/java/com/deepecg/app/
│   ├── MainActivity.kt      UI : photo/galerie, affichage du résultat
│   ├── EcgDigitizer.kt      photo -> signal ECG (mV)
│   └── EcgClassifier.kt     signal -> classe (TFLite embarqué)
├── app/src/main/assets/     ecg_model.tflite + model_info.txt (à générer)
└── app/src/main/res/        layouts, chaînes (FR), icônes
```

## Limites

- La numérisation d'une photo est une heuristique : photo floue, reflet,
  tracé très dense ou papier non millimétré dégradent le résultat.
- Le modèle classe des enregistrements de 30 s à 300 Hz ; une photo couvrant
  une durée différente est rééchantillonnée sur 30 s.
- Précision dépend entièrement du modèle entraîné et de la qualité de la
  photo.
