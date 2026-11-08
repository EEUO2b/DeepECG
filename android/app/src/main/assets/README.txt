Placez ici le modèle exporté, depuis la racine du dépôt :

    python export_tflite.py --pretrained andreotti   # 1 dérivation, 30 s @ 300 Hz (CinC 2017, GPL v3)
    python export_tflite.py --pretrained ptbxl       # 12 dérivations, 10 s @ 100 Hz (PTB-XL, MIT)
    python export_tflite.py --model Keras_models/VOTRE_MODELE.h5 --arch cnn

Cela génère dans ce dossier :
  - ecg_model.tflite  (modèle principal TensorFlow Lite)
  - model_info.txt    (forme d'entrée, nombre de dérivations, normalisation, classes)
  - af_model.tflite + af_model_info.txt  (modèle FA 1 dérivation, mode ptbxl uniquement :
    lancé sur la dérivation II pour une probabilité de FA dédiée)

Sans ecg_model.tflite, l'application affiche un message d'erreur à l'ouverture.
