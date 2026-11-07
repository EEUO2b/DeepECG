"""Export a trained DeepECG Keras model to TensorFlow Lite for the Android app.

Usage (from the repository root, with TensorFlow 2.x installed):

    python export_tflite.py --model Keras_models/my_model.h5 --arch cnn

This writes ``ecg_model.tflite`` and ``model_info.txt`` into
``android/app/src/main/assets/``. Copy the folder ``android/`` to your machine
with Android Studio and run the app on a device (minSdk 24).

``--arch cnn``    : CNN_ECG.py model, input (1, 20, 450, 1), raw values in mV.
``--arch conv1d`` : Conv1D_ECG.py model, input (1, 10100, 1), values must be
                    z-score normalized per segment (as in Conv1D_ECG.py);
                    the Android app reads ``normalize=true`` from model_info.txt
                    and applies the same normalization automatically.

Classes (PhysioNet/CinC 2017): 0=N Normal, 1=A Atrial fibrillation,
2=O Other, 3=~ Noisy.
"""

import argparse
import os
import sys

DEFAULT_OUTPUT = os.path.join('android', 'app', 'src', 'main', 'assets', 'ecg_model.tflite')


def main():
    parser = argparse.ArgumentParser(description='Export a DeepECG Keras model to TFLite.')
    parser.add_argument('--model', required=True, help='Path to the trained Keras model (.h5 or .keras)')
    parser.add_argument('--arch', choices=['cnn', 'conv1d'], default='cnn',
                        help='Which training script produced the model (default: cnn)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Destination .tflite file')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit('Model file not found: %s' % args.model)

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    input_shape = tuple(model.input_shape)
    print('Keras input shape:', input_shape)

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

    dims = [d for d in input_shape if d is not None]
    length = 1
    for d in dims[1:]:  # skip batch dimension
        length *= d
    normalize = 'true' if args.arch == 'conv1d' else 'false'
    info = (
        'input_shape=%s\n' % ','.join(str(d) for d in input_shape if d is not None)
        + 'input_length=%d\n' % length
        + 'normalize=%s\n' % normalize
        + 'labels=N,A,O,~\n'
    )
    info_path = os.path.join(out_dir or '.', 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write(info)
    print('Wrote %s' % info_path)
    print('input_length=%d normalize=%s' % (length, normalize))


if __name__ == '__main__':
    main()
