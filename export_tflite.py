"""Export a trained ECG classification model to TensorFlow Lite for the Android app.

Two sources are supported:

1) Your own model trained with this repository:

    python export_tflite.py --model Keras_models/my_model.h5 --arch cnn

   ``--arch cnn``    : CNN_ECG.py model, input (1, 20, 450, 1), raw values in mV,
                       labels N,A,O,~ (no normalization).
   ``--arch conv1d`` : Conv1D_ECG.py model, input (1, 10100, 1), z-score normalized
                       per segment, labels N,A,O,~.

2) A ready-made pre-trained model compatible with the app (downloaded on demand,
   not committed to this repo):

    python export_tflite.py --pretrained andreotti

   Downloads ``ResNet_30s_34lay_16conv.hdf5`` (ResNet-34, 30 s @ 300 Hz segments,
   input (1, 9000, 1), z-score normalized, labels A,N,O,~) from the PhysioNet/CinC
   Challenge 2017 entry of Andreotti et al.:
   https://github.com/fernandoandreotti/cinc-challenge2017 (license: GPL v3).
   Citation: Andreotti F., Carr O., Pimentel M.A.F., Mahdi A., De Vos M. (2017).
   "Comparing Feature Based Classifiers and Convolutional Neural Networks to
   Detect Arrhythmia from Short Segments of ECG". Computing in Cardiology.

Both modes write ``ecg_model.tflite`` and ``model_info.txt`` into
``android/app/src/main/assets/``. Requires TensorFlow 2.x.
"""

import argparse
import os
import sys
import urllib.request

DEFAULT_OUTPUT = os.path.join('android', 'app', 'src', 'main', 'assets', 'ecg_model.tflite')

PRETRAINED = {
    'andreotti': {
        'url': 'https://raw.githubusercontent.com/fernandoandreotti/cinc-challenge2017/master/deeplearn-approach/ResNet_30s_34lay_16conv.hdf5',
        'file': 'ResNet_30s_34lay_16conv.hdf5',
        'labels': 'A,N,O,~',
        'normalize': 'true',
        'source': 'https://github.com/fernandoandreotti/cinc-challenge2017 (PhysioNet/CinC Challenge 2017 entry, Andreotti et al. 2017)',
        'license': 'GPL-3.0',
    }
}

PRETRAINED_DIR = 'pretrained'


def download_pretrained(name):
    spec = PRETRAINED[name]
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    path = os.path.join(PRETRAINED_DIR, spec['file'])
    if not os.path.isfile(path):
        print('Downloading %s -> %s' % (spec['url'], path))
        urllib.request.urlretrieve(spec['url'], path)
    else:
        print('Already present: %s' % path)
    return path, spec


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

    import tensorflow as tf

    provenance = None
    if args.pretrained:
        model_path, spec = download_pretrained(args.pretrained)
        labels = spec['labels']
        normalize = spec['normalize']
        provenance = spec
    else:
        model_path = args.model
        if not os.path.isfile(model_path):
            sys.exit('Model file not found: %s' % model_path)
        labels = 'N,A,O,~'
        normalize = 'true' if args.arch == 'conv1d' else 'false'

    # compile=False: legacy Keras 2.x files carry an old optimizer config that
    # recent Keras refuses to deserialize; we only need the weights for inference.
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = tuple(model.input_shape)
    print('Keras input shape:', input_shape)

    if args.pretrained is None:
        expected = (None, 20, 450, 1) if args.arch == 'cnn' else (None, 10100, 1)
        if input_shape != expected:
            print('WARNING: expected input shape %s for --arch %s, got %s'
                  % (expected, args.arch, input_shape))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(tflite_model)
    print('Wrote %s (%d bytes)' % (args.output, len(tflite_model)))

    dims = [d for d in input_shape[1:] if d is not None]  # skip batch dimension
    length = 1
    for d in dims:
        length *= d

    lines = [
        'input_shape=%s' % ','.join(str(d) for d in input_shape if d is not None),
        'input_length=%d' % length,
        'normalize=%s' % normalize,
        'labels=%s' % labels,
    ]
    if provenance:
        lines += [
            'source=%s' % provenance['source'],
            'license=%s' % provenance['license'],
        ]
    info_path = os.path.join(out_dir or '.', 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote %s' % info_path)
    print('input_length=%d normalize=%s labels=%s' % (length, normalize, labels))


if __name__ == '__main__':
    main()
