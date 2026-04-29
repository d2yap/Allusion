# https://huggingface.co/spaces/SmilingWolf/wd-tagger

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

DEFAULT_MODEL_FILENAMES = ("model.onnx", "model.pt", "model.pth", "model.h5")
DEFAULT_LABEL_FILENAMES = ("selected_tags.csv", "tags.csv", "tags.txt", "names.txt")

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()


def load_labels(dataframe) -> Tuple[List[str], List[int], List[int], List[int]]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def load_labels_from_path(path: str) -> Tuple[List[str], List[int], List[int], List[int]]:
    lower = path.lower()
    if lower.endswith(".csv"):
        dataframe = pd.read_csv(path)
        if "name" not in dataframe.columns:
            if "tag" in dataframe.columns:
                dataframe = dataframe.rename(columns={"tag": "name"})
            elif "label" in dataframe.columns:
                dataframe = dataframe.rename(columns={"label": "name"})
        if "category" not in dataframe.columns:
            dataframe["category"] = 0
        return load_labels(dataframe)

    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file_handle:
            entries = json.load(file_handle)
        names = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                names.append(str(entry.get("name") or entry.get("tag") or entry.get("label") or ""))
        dataframe = pd.DataFrame({"name": names, "category": [0] * len(names)})
        return load_labels(dataframe)

    with open(path, "r", encoding="utf-8") as file_handle:
        names = [line.strip() for line in file_handle if line.strip()]
    dataframe = pd.DataFrame({"name": names, "category": [0] * len(names)})
    return load_labels(dataframe)


def resolve_first_existing_file(paths: Iterable[str]) -> str:
    for candidate in paths:
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def resolve_model_and_taglist(
    project_path: Optional[str],
    model_path: Optional[str],
    taglist_path: Optional[str],
) -> Tuple[str, str]:
    if model_path and taglist_path:
        return model_path, taglist_path

    if not project_path:
        return model_path or "", taglist_path or ""

    resolved_model = resolve_first_existing_file(
        os.path.join(project_path, filename) for filename in DEFAULT_MODEL_FILENAMES
    )
    resolved_taglist = resolve_first_existing_file(
        os.path.join(project_path, filename) for filename in DEFAULT_LABEL_FILENAMES
    )
    return model_path or resolved_model, taglist_path or resolved_taglist


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None

    def download_model(self, model_repo):
        raise RuntimeError("download_model removed: this Predictor only supports local files")

    def load_model_from_files(self, model_path: str, csv_path: str):
        # Load taglist CSV and ONNX model from local paths only.
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Taglist not found: {csv_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        key = f"{model_path}:{csv_path}"
        if key == self.last_loaded_repo:
            return

        sep_tags = load_labels_from_path(csv_path)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        del self.model
        model = rt.InferenceSession(model_path)
        try:
            _, height, width, _ = model.get_inputs()[0].shape
            self.model_target_size = height
        except Exception:
            self.model_target_size = 512

        self.last_loaded_repo = key
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        model_path,
        taglist_path,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        # Load local files
        self.load_model_from_files(model_path, taglist_path)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings)

        return sorted_general_strings, rating, character_res, general_res


def build_tags_payload(general_res, character_res):
    merged = {}
    merged.update(general_res)
    merged.update(character_res)
    return [
        {"tag": tag_name, "score": float(score)}
        for tag_name, score in sorted(merged.items(), key=lambda item: item[1], reverse=True)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--project', help='Path to a local WD14 project directory')
    parser.add_argument('--model', help='Path to ONNX or Torch model file')
    parser.add_argument('--taglist', help='Path to taglist CSV, JSON, or text file')
    parser.add_argument('--threshold', type=float, default=None, help='Apply one threshold to both general and character tags')
    parser.add_argument('--general-thresh', type=float, default=0.35)
    parser.add_argument('--character-thresh', type=float, default=0.85)
    parser.add_argument('--general-mcut', action='store_true')
    parser.add_argument('--character-mcut', action='store_true')
    args = parser.parse_args()

    model_path, taglist_path = resolve_model_and_taglist(args.project, args.model, args.taglist)
    if not model_path:
        print(json.dumps({"tags": [], "error": "Model file not found. Pass --model or --project."}, ensure_ascii=False))
        return
    if not taglist_path:
        print(json.dumps({"tags": [], "error": "Tag list file not found. Pass --taglist or --project."}, ensure_ascii=False))
        return

    if args.threshold is not None:
        args.general_thresh = args.threshold
        args.character_thresh = args.threshold

    predictor = Predictor()

    # load image
    img = Image.open(args.image).convert('RGBA')

    gen_str, rating, char_res, gen_res = predictor.predict(
        img,
        model_path,
        taglist_path,
        args.general_thresh,
        args.general_mcut,
        args.character_thresh,
        args.character_mcut,
    )

    out = {
        'tags': build_tags_payload(gen_res, char_res),
        'general_string': gen_str,
        'rating': rating,
        'characters': char_res,
        'general': gen_res,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
