Placez ici le modèle exporté, depuis la racine du dépôt :

    python export_tflite.py --model Keras_models/VOTRE_MODELE.h5

Cela génère dans ce dossier :
  - ecg_model.tflite  (modèle TensorFlow Lite)
  - model_info.txt    (forme d'entrée, normalisation, classes)

Sans ecg_model.tflite, l'application affiche un message d'erreur à l'ouverture.
