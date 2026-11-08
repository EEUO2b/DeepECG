"""Export a trained ECG classification model to TensorFlow Lite for the Android app.

Three sources are supported:

1) Your own model trained with this repository:

    python export_tflite.py --model Keras_models/my_model.h5 --arch cnn

   ``--arch cnn``    : CNN_ECG.py model, input (1, 20, 450, 1), raw values in mV,
                       labels N,A,O,~ (no normalization).
   ``--arch conv1d`` : Conv1D_ECG.py model, input (1, 10100, 1), z-score normalized
                       per segment, labels N,A,O,~.

2) Ready-made single-lead model (PhysioNet/CinC 2017, downloaded on demand):

    python export_tflite.py --pretrained andreotti

   ResNet-34, input (1, 9000, 1) = 30 s @ 300 Hz, z-score normalized,
   labels A,N,O,~. Source: https://github.com/fernandoandreotti/cinc-challenge2017
   (license GPL v3; Andreotti et al., Computing in Cardiology 2017).

3) Ready-made 12-lead model (PTB-XL, downloaded on demand):

    python export_tflite.py --pretrained ptbxl

   CNN from https://huggingface.co/Steenslid/ecg-ptbxl-classification (license MIT),
   input (1, 1000, 12) = 10 s @ 100 Hz, standard 12-lead order, per-lead z-score
   normalization baked into the exported TFLite, multi-label output over the 5
   PTB-XL superclasses NORM, MI, STTC, CD, HYP (with decision thresholds).
   Dataset: PTB-XL (Wagner et al., Scientific Data 2020).
   This mode ALSO exports the single-lead Andreotti ResNet as ``af_model.tflite``
   (+ ``af_model_info.txt``): the app runs it on lead II to give a dedicated
   atrial-fibrillation probability (the PTB-XL superclasses have no AF class).

All modes write ``ecg_model.tflite`` and ``model_info.txt`` into
``android/app/src/main/assets/``. Requires TensorFlow 2.x and numpy.
"""

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_OUTPUT = os.path.join('android', 'app', 'src', 'main', 'assets', 'ecg_model.tflite')
PRETRAINED_DIR = 'pretrained'

PRETRAINED = {
    'andreotti': {
        'kind': 'keras',
        'files': {
            'model': ('https://raw.githubusercontent.com/fernandoandreotti/cinc-challenge2017/master/deeplearn-approach/ResNet_30s_34lay_16conv.hdf5',
                      'ResNet_30s_34lay_16conv.hdf5'),
        },
        'labels': 'A,N,O,~',
        'normalize': 'true',
        'sampling_hz': 300,
        'source': 'https://github.com/fernandoandreotti/cinc-challenge2017 (PhysioNet/CinC Challenge 2017 entry, Andreotti et al. 2017)',
        'license': 'GPL-3.0',
    },
    'ptbxl': {
        'kind': 'ptbxl',
        'files': {
            'model': ('https://huggingface.co/Steenslid/ecg-ptbxl-classification/resolve/main/ecg_model.keras', 'ptbxl_ecg_model.keras'),
            'params': ('https://huggingface.co/Steenslid/ecg-ptbxl-classification/resolve/main/normalisation_params.npz', 'ptbxl_normalisation_params.npz'),
            'thresholds': ('https://huggingface.co/Steenslid/ecg-ptbxl-classification/resolve/main/thresholds.json', 'ptbxl_thresholds.json'),
        },
        'labels': 'NORM,MI,STTC,CD,HYP',
        'labels_fr': 'Normal,Infarctus du myocarde,Modifications ST-T,Troubles de conduction,Hypertrophie',
        'multilabel': 'true',
        'normalize': 'false',
        'sampling_hz': 100,
        'source': 'https://huggingface.co/Steenslid/ecg-ptbxl-classification (12-lead CNN on PTB-XL, Steenslid & Kvamme; dataset PTB-XL, Wagner et al. 2020 Sci Data)',
        'license': 'MIT',
    },
}


def download(url, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print('Already present: %s' % dest)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print('Downloading %s -> %s' % (url, dest))
    urllib.request.urlretrieve(url, dest)
    return dest


def export_ptbxl(spec, output):
    """Fuse per-lead z-score normalization into the PTB-XL CNN and convert to TFLite."""
    import numpy as np
    import tensorflow as tf

    paths = {}
    for key, (url, fname) in spec['files'].items():
        paths[key] = download(url, os.path.join(PRETRAINED_DIR, 'ptbxl', fname))

    model = tf.keras.models.load_model(paths['model'], compile=False)
    params = np.load(paths['params'])
    mean = params['mean'].reshape(-1).astype(np.float32)
    std = params['std'].reshape(-1).astype(np.float32)
    with open(paths['thresholds']) as f:
        thr = json.load(f)
    labels = spec['labels'].split(',')
    thresholds = ','.join(str(thr[c]) for c in labels)

    inp = tf.keras.Input(shape=model.input_shape[1:], name='ecg_mv')
    xn = (inp - tf.constant(mean, dtype=tf.float32)) / tf.constant(std, dtype=tf.float32)
    fused = tf.keras.Model(inp, model(xn))

    converter = tf.lite.TFLiteConverter.from_keras_model(fused)
    tflite_model = converter.convert()
    write_output(output, tflite_model)

    shape = tuple(fused.input_shape)  # (None, 1000, 12)
    samples_per_lead = shape[1]
    num_leads = shape[2]
    info = {
        'input_shape': '1,%d,%d' % (samples_per_lead, num_leads),
        'input_length': samples_per_lead * num_leads,
        'num_leads': num_leads,
        'samples_per_lead': samples_per_lead,
        'sampling_hz': spec['sampling_hz'],
        'normalize': spec['normalize'],
        'multilabel': spec['multilabel'],
        'labels': spec['labels'],
        'labels_fr': spec['labels_fr'],
        'thresholds': thresholds,
        'source': spec['source'],
        'license': spec['license'],
    }
    write_info(output, info)
    print('input=%s num_leads=%d samples_per_lead=%d multilabel=%s labels=%s'
          % (info['input_shape'], num_leads, samples_per_lead, spec['multilabel'], spec['labels']))
    return info


def export_keras_model(model_path, output, labels, normalize, sampling_hz, source=None, license=None, extra=None, info_name='model_info.txt'):
    import tensorflow as tf

    # compile=False: legacy Keras 2.x files carry an old optimizer config that
    # recent Keras refuses to deserialize; we only need the weights for inference.
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = tuple(model.input_shape)
    print('Keras input shape:', input_shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    write_output(output, tflite_model)

    dims = [d for d in input_shape[1:] if d is not None]
    length = 1
    for d in dims:
        length *= d
    num_leads = dims[-1] if len(dims) >= 2 else 1
    samples_per_lead = length // num_leads
    info = {
        'input_shape': ','.join(str(d) for d in input_shape if d is not None),
        'input_length': length,
        'num_leads': num_leads,
        'samples_per_lead': samples_per_lead,
        'sampling_hz': sampling_hz,
        'normalize': normalize,
        'multilabel': 'false',
        'labels': labels,
    }
    if source:
        info['source'] = source
        info['license'] = license
    if extra:
        info.update(extra)
    write_info(output, info, info_name)
    print('input_length=%d num_leads=%d normalize=%s labels=%s' % (length, num_leads, normalize, labels))
    return info


def write_output(output, tflite_model):
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, 'wb') as f:
        f.write(tflite_model)
    print('Wrote %s (%d bytes)' % (output, len(tflite_model)))


def write_info(output, info, info_name='model_info.txt'):
    info_path = os.path.join(os.path.dirname(output) or '.', info_name)
    with open(info_path, 'w') as f:
        for k, v in info.items():
            f.write('%s=%s\n' % (k, v))
    print('Wrote %s' % info_path)


def main():
    parser = argparse.ArgumentParser(description='Export an ECG classification model to TFLite for the Android app.')
    parser.add_argument('--model', help='Path to your own trained Keras model (.h5 or .keras)')
    parser.add_argument('--arch', choices=['cnn', 'conv1d'], default='cnn',
                        help='Which training script produced --model (default: cnn)')
    parser.add_argument('--pretrained', choices=sorted(PRETRAINED.keys()),
                        help='Download and convert a ready-made pre-trained model instead of --model')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Destination .tflite file')
    args = parser.parse_args()

    if bool(args.model) == bool(args.pretrained):
        sys.exit('Give exactly one of --model or --pretrained.')

    if args.pretrained:
        spec = PRETRAINED[args.pretrained]
        if spec['kind'] == 'ptbxl':
            export_ptbxl(spec, args.output)
            # secondary single-lead AF model, run by the app on lead II
            af_spec = PRETRAINED['andreotti']
            af_path = download(af_spec['files']['model'][0],
                               os.path.join(PRETRAINED_DIR, af_spec['files']['model'][1]))
            af_output = os.path.join(os.path.dirname(args.output) or '.', 'af_model.tflite')
            export_keras_model(af_path, af_output, af_spec['labels'], af_spec['normalize'],
                               af_spec['sampling_hz'], af_spec['source'], af_spec['license'],
                               extra={'pad_short': 'true'}, info_name='af_model_info.txt')
            print('Also exported AF model -> %s' % af_output)
        else:
            path = download(spec['files']['model'][0],
                            os.path.join(PRETRAINED_DIR, spec['files']['model'][1]))
            export_keras_model(path, args.output, spec['labels'], spec['normalize'],
                               spec['sampling_hz'], spec['source'], spec['license'])
    else:
        if not os.path.isfile(args.model):
            sys.exit('Model file not found: %s' % args.model)
        labels = 'N,A,O,~'
        normalize = 'true' if args.arch == 'conv1d' else 'false'
        expected = (None, 20, 450, 1) if args.arch == 'cnn' else (None, 10100, 1)
        export_keras_model(args.model, args.output, labels, normalize, 300)
        print('Note: expected input shape %s for --arch %s' % (expected, args.arch))


if __name__ == '__main__':
    main()
