"""
Convert a trained Keras .h5 model to TensorFlow Lite (.tflite) for
Raspberry Pi deployment.

Usage (run on your desktop/laptop where TensorFlow is installed):
    python convert_model_to_tflite.py

This reads  models/best_model.h5
and writes  models/best_model.tflite
"""

import os
import sys
import numpy as np

def convert(h5_path: str = "models/best_model.h5",
            tflite_path: str = "models/best_model.tflite"):
    import tensorflow as tf
    from models.model_architecture import FocalLoss

    print(f"Loading Keras model from {h5_path} ...")
    model = tf.keras.models.load_model(
        h5_path,
        custom_objects={'FocalLoss': FocalLoss}
    )
    model.summary()

    print("Converting to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # representative dataset for full integer quantisation (optional)
    def representative_dataset():
        for _ in range(100):
            yield [np.random.randn(1, 216, 1).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path) or '.', exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Saved TFLite model to {tflite_path}  ({size_kb:.0f} KB)")
    print("Copy this file to your Raspberry Pi's models/ folder.")


if __name__ == "__main__":
    h5 = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.h5"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.tflite"
    convert(h5, out)
