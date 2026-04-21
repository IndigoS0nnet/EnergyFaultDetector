import os

# Must be set before any `import tensorflow`. Keeps Keras 2 semantics via the
# tf-keras shim so the `.model`/`.encoder` save format used by Autoencoder.save
# keeps working on TF 2.16+ (which defaults to Keras 3).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
