#!/usr/bin/env python3
"""
DeepDanbooru prediction helper (TensorFlow).

This script attempts to load a Keras .h5 model from the provided project directory
and predict tags for a given image. Output is a single JSON object printed to
stdout. On success the JSON has the shape:
  {"tags": [{"tag": "tag_name", "score": 0.95}, ...]}

Usage:
  python predict.py --image "C:/path/to/image.jpg" --project "C:/path/to/deepdanbooru/project" --threshold 0.5

Assumptions / conventions:
- A single .h5 model file exists in the project directory (first .h5 file will be used).
- A tags file exists in the project directory named one of: tags.txt, names.txt (one tag per line, in model order).
- The model accepts an input shape compatible with an RGB image; the script will
  attempt to infer the required height and width from the model input shape.

If TensorFlow or Pillow are not available, the script will return a JSON error
with installation instructions.
"""
import argparse
import json
import sys
import os
import glob
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--project', required=False)
parser.add_argument('--model', required=False, help='Path to a specific .h5 model file')
parser.add_argument('--tag-list', required=False, help='Path to a tag list file (txt/json)')
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()


def err_json(message: str, exc: Exception = None):
  payload = {"tags": [], "error": message}
  if exc is not None:
    payload["python_error"] = str(exc)
  print(json.dumps(payload))
  sys.exit(0)


try:
  from PIL import Image
except Exception as e:
  err_json(
    "Pillow (PIL) is required but not available. Install it with: python -m pip install pillow",
    e,
  )

try:
  # Reduce TensorFlow logging (info/warning) which otherwise may appear on stderr.
  import os
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
  import tensorflow as tf
  from tensorflow.keras.models import load_model
except Exception as e:
  err_json(
    "TensorFlow is required but not available. Install a compatible tensorflow package in your Python environment (e.g. python -m pip install tensorflow)",
    e,
  )


def find_model(project_path: str) -> str:
  # Prefer explicit model.h5, otherwise take the first .h5
  candidates = [
    os.path.join(project_path, 'model.h5'),
  ]
  candidates += glob.glob(os.path.join(project_path, '*.h5'))
  for c in candidates:
    if os.path.isfile(c):
      return c
  return ''


def find_tags_file(project_path: str) -> str:
  names = ['tags.txt', 'names.txt', 'tags.csv']
  for n in names:
    p = os.path.join(project_path, n)
    if os.path.isfile(p):
      return p
  # fallback: any .txt in project dir
  txts = glob.glob(os.path.join(project_path, '*.txt'))
  return txts[0] if txts else ''


def load_tags(path: str) -> List[str]:
  with open(path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
  return lines


def preprocess_image(img_path: str, target_size):
  img = Image.open(img_path).convert('RGB')
  img = img.resize(target_size, Image.LANCZOS)
  import numpy as np

  arr = np.asarray(img).astype('float32') / 255.0
  # model expects shape (batch, h, w, 3)
  arr = arr[np.newaxis, ...]
  return arr


def main():
  image_path = args.image
  project_path = args.project
  explicit_model = args.model
  explicit_taglist = args.tag_list if hasattr(args, 'tag_list') else None
  threshold = float(args.threshold)

  if not os.path.isfile(image_path):
    err_json(f'Image not found: {image_path}')

  # Validate image
  if not os.path.isfile(image_path):
    err_json(f'Image not found: {image_path}')

  # Determine model path
  model_path = ''
  if explicit_model:
    if os.path.isfile(explicit_model):
      model_path = explicit_model
    else:
      err_json(f'Model file not found: {explicit_model}')

  # If no explicit model, fall back to project directory lookup
  if model_path == '':
    if not project_path or not os.path.isdir(project_path):
      err_json(f'Project directory not found: {project_path}')
    model_path = find_model(project_path)
    if model_path == '':
      err_json('No .h5 model file found in project directory. Place your pretrained .h5 model in the project folder or pass --model /path/to/model.h5')

  # Determine tags file: prefer explicit tag-list, then project dir (or model dir)
  tags_file = ''
  if explicit_taglist:
    if os.path.isfile(explicit_taglist):
      tags_file = explicit_taglist
    else:
      err_json(f'Tag list file not found: {explicit_taglist}')

  if tags_file == '':
    # If project path not provided but model was explicit, look for tags next to the model
    search_dir = project_path if project_path and os.path.isdir(project_path) else os.path.dirname(model_path)
    tags_file = find_tags_file(search_dir)
    if tags_file == '':
      err_json('Could not find a tags file (tags.txt / names.txt) in the project or model directory. You can pass --tag-list /path/to/tags.txt')

  try:
    tags = load_tags(tags_file)
  except Exception as e:
    err_json('Failed to read tags file.', e)

  try:
    model = load_model(model_path, compile=False)
  except Exception as e:
    err_json('Failed to load model. Make sure the .h5 was saved with a compatible TensorFlow/Keras version and any custom objects are available.', e)

  # Infer input size from model
  try:
    input_shape = model.input_shape  # e.g. (None, H, W, 3)
    # normalize to (H, W)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
      # handle (None, H, W, C) or (None, C, H, W)
      if input_shape[1] in (1, 3) and len(input_shape) >= 4:
        # channel-first (None, C, H, W)
        h = int(input_shape[2])
        w = int(input_shape[3])
      else:
        h = int(input_shape[1])
        w = int(input_shape[2])
    else:
      h = 512
      w = 512
  except Exception:
    h = 512
    w = 512

  try:
    arr = preprocess_image(image_path, (w, h))
  except Exception as e:
    err_json('Failed to open or preprocess image.', e)

  try:
    # Disable Keras progress output with verbose=0 so stdout contains only our JSON result
    preds = model.predict(arr, verbose=0)
    # preds could be (1, N) or (N,)
    import numpy as np

    if isinstance(preds, list):
      # Keras model with multiple outputs is not supported by this helper
      err_json('Model returned multiple outputs; this helper expects a single vector output.')
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
      scores = preds[0]
    elif preds.ndim == 1:
      scores = preds
    else:
      # Unexpected shape
      scores = preds.flatten()
  except Exception as e:
    err_json('Model prediction failed.', e)

  if len(scores) != len(tags):
    # return all scores but warn that lengths mismatch
    # still map up to min length
    mapped = [
      {"tag": tags[i], "score": float(scores[i])}
      for i in range(min(len(tags), len(scores)))
    ]
    print(json.dumps({"tags": mapped, "error": f"Tags count ({len(tags)}) and model outputs ({len(scores)}) mismatch. Mapped up to min length."}))
    sys.exit(0)

  mapped = []
  for t, s in zip(tags, scores):
    mapped.append({"tag": t, "score": float(s)})

  # Filter by threshold here as a convenience; the app also filters client-side.
  mapped = [m for m in mapped if m["score"] >= threshold]

  print(json.dumps({"tags": mapped}))


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    err_json('Unexpected error while running prediction.', e)
