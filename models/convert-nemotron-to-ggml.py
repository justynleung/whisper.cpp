#!/usr/bin/env python3
# Convert Nemotron 3.5 ASR from NeMo format to a ggml-compatible format.

import argparse
import os
import re
import struct
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


GGML_MAGIC = 0x67676d6c
NEMOTRON_FILE_VERSION = 1
NEMOTRON_EXPECTED_TENSORS = 657
NEMOTRON_VOCAB_SIZE = 13088
NEMOTRON_BLANK_ID = NEMOTRON_VOCAB_SIZE - 1
NEMOTRON_ENCODER_LAYERS = 24
NEMOTRON_PRED_LAYERS = 2
SPECIAL_TOKEN_FIELDS = ("blank", "unk", "bos", "sos", "eos", "eot")


def safe_extract_nemo_archive(nemo_path, extract_dir):
    print(f"Extracting {nemo_path} to {extract_dir}")

    extract_root = Path(extract_dir).resolve()
    with tarfile.open(nemo_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (extract_root / member.name).resolve()
            if not str(target).startswith(str(extract_root) + os.sep):
                raise ValueError(f"Refusing to extract path outside target dir: {member.name}")
        tar.extractall(path=extract_dir)

    print("Extraction complete")


def load_model_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def parse_arch_tensor_names(header_path):
    text = Path(header_path).read_text(encoding="utf-8")

    map_block = re.search(r"NEMOTRON_TENSOR_NAMES\s*=\s*\{(.*?)\};", text, re.DOTALL)
    if not map_block:
        raise ValueError(f"Could not find NEMOTRON_TENSOR_NAMES in {header_path}")

    templates = []
    for match in re.finditer(r'\{\s*\w+\s*,\s*"([^"]+)"\s*\}', map_block.group(1)):
        templates.append(match.group(1))

    if not templates:
        raise ValueError(f"No tensor names found in {header_path}")

    names = []
    for template in templates:
        if "%d" in template:
            if template.startswith("encoder.layers."):
                names.extend(template.replace("%d", str(i)) for i in range(NEMOTRON_ENCODER_LAYERS))
            elif template.startswith("decoder.prediction.dec_rnn.lstm."):
                names.extend(template.replace("%d", str(i)) for i in range(NEMOTRON_PRED_LAYERS))
            else:
                raise ValueError(f"Do not know how many times to expand tensor template: {template}")
        else:
            names.append(template)

    if len(names) != NEMOTRON_EXPECTED_TENSORS:
        raise ValueError(
            f"Expected {NEMOTRON_EXPECTED_TENSORS} tensor names from arch header, got {len(names)}"
        )

    return names


def find_required_file(extract_dir, filename):
    matches = sorted(Path(extract_dir).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} in extracted .nemo archive")
    if len(matches) > 1:
        raise ValueError(f"Found multiple {filename} files in extracted .nemo archive")
    return matches[0]


def find_nemo_resource(extract_dir, resource):
    if not isinstance(resource, str):
        return None

    name = resource.removeprefix("nemo:")
    matches = sorted(Path(extract_dir).rglob(name))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Found multiple tokenizer resource files named {name}")
    return matches[0]


def load_vocabulary(config):
    vocabulary = config.get("joint", {}).get("vocabulary")
    if not vocabulary:
        raise ValueError("Missing joint.vocabulary in model_config.yaml")

    tokens = [str(token) for token in vocabulary]
    if len(tokens) + 1 != NEMOTRON_VOCAB_SIZE:
        raise ValueError(
            f"Expected {NEMOTRON_VOCAB_SIZE - 1} tokenizer tokens plus blank, got {len(tokens)}"
        )

    tokens.append("<blank>")
    return tokens


def special_token_ids(tokens):
    def token_id(*candidates):
        for candidate in candidates:
            if candidate in tokens:
                return tokens.index(candidate)
        return -1

    ids = {
        "blank": NEMOTRON_BLANK_ID,
        "unk": token_id("<unk>", "<UNK>", "unk"),
        "bos": token_id("<s>", "<bos>", "<BOS>"),
        "sos": token_id("<sos>", "<SOS>"),
        "eos": token_id("</s>", "<eos>", "<EOS>"),
        "eot": token_id("<|endoftext|>", "<eot>", "<EOT>"),
    }

    if ids["unk"] < 0:
        raise ValueError("Could not resolve unknown-token id")

    return ids


def validate_hparams(config):
    preprocessor = config["preprocessor"]
    encoder = config["encoder"]
    decoder = config["decoder"]
    prednet = decoder["prednet"]
    model_defaults = config["model_defaults"]
    decoding = config.get("decoding", {}).get("greedy", {})

    hparams = {
        "sample_rate": int(config["sample_rate"]),
        "n_fft": int(preprocessor["n_fft"]),
        "window_size": int(round(float(preprocessor["window_size"]) * int(config["sample_rate"]))),
        "hop_length": int(round(float(preprocessor["window_stride"]) * int(config["sample_rate"]))),
        "n_mels": int(preprocessor["features"]),
        "subsampling_factor": int(encoder["subsampling_factor"]),
        "n_subsampling_channels": int(encoder["subsampling_conv_channels"]),
        "n_conv_kernel": int(encoder["conv_kernel_size"]),
        "n_audio_state": int(encoder["d_model"]),
        "n_audio_head": int(encoder["n_heads"]),
        "n_audio_layer": int(encoder["n_layers"]),
        "n_pred_dim": int(prednet["pred_hidden"]),
        "n_pred_layers": int(prednet["pred_rnn_layers"]),
        "n_prompts": int(model_defaults["num_prompts"]),
        "n_vocab": NEMOTRON_VOCAB_SIZE,
        "n_max_tokens": int(decoding.get("max_symbols", 10)),
    }

    expected = {
        "sample_rate": 16000,
        "n_fft": 512,
        "window_size": 400,
        "hop_length": 160,
        "n_mels": 128,
        "subsampling_factor": 8,
        "n_audio_state": 1024,
        "n_audio_head": 8,
        "n_audio_layer": 24,
        "n_pred_dim": 640,
        "n_pred_layers": 2,
        "n_prompts": 128,
        "n_vocab": 13088,
    }
    mismatches = [
        f"{key}: expected {expected_value}, got {hparams[key]}"
        for key, expected_value in expected.items()
        if hparams[key] != expected_value
    ]
    if mismatches:
        raise ValueError("Incompatible Nemotron hparams:\n  " + "\n  ".join(mismatches))

    return hparams


def validate_architecture_tensors(state_dict, expected_names):
    checkpoint_names = set(state_dict.keys())
    expected = set(expected_names)

    missing = sorted(expected - checkpoint_names)
    extra = sorted(checkpoint_names - expected)
    forbidden = sorted(
        name
        for name in checkpoint_names
        if "num_batches_tracked" in name
        or "running_mean" in name
        or "running_var" in name
        or "tdt" in name.lower()
        or "duration" in name.lower()
    )

    errors = []
    if missing:
        errors.append("Missing expected tensors:\n  " + "\n  ".join(missing))
    if extra:
        errors.append("Unexpected architectural tensors:\n  " + "\n  ".join(extra))
    if forbidden:
        errors.append("Forbidden Parakeet/TDT-style tensors:\n  " + "\n  ".join(forbidden))

    if errors:
        raise ValueError("\n\n".join(errors))

    if len(expected_names) != NEMOTRON_EXPECTED_TENSORS:
        raise ValueError(f"Expected {NEMOTRON_EXPECTED_TENSORS} tensor names, got {len(expected_names)}")


def tensor_to_numpy(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().cpu().contiguous().numpy()


def converted_tensor_data(name, tensor):
    data = tensor_to_numpy(tensor)

    # ggml conv code expects convolution weights to retain their full spatial layout.
    if not ("conv" in name and "weight" in name):
        data = np.squeeze(data)

    return np.ascontiguousarray(data)


def tensor_ftype(name, data, use_f16):
    if not use_f16:
        return 0, data.astype(np.float32)

    if (
        data.ndim < 2
        or "bias" in name
        or "norm" in name
        or "preprocessor.featurizer" in name
        or ("conv" in name and data.ndim >= 3)
    ):
        return 0, data.astype(np.float32)

    return 1, data.astype(np.float16)


def write_tensor(fout, name, data, use_f16):
    ftype, data = tensor_ftype(name, data, use_f16)
    n_dims = data.ndim
    name_bytes = name.encode("utf-8")

    print(f"  {name}: shape={list(data.shape)} dtype={data.dtype}")

    fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(name_bytes)
    data.tofile(fout)


def write_string(fout, text):
    data = text.encode("utf-8")
    fout.write(struct.pack("i", len(data)))
    fout.write(data)


def write_model_file(output_path, hparams, tokens, special_ids, state_dict, expected_names, use_f16):
    with open(output_path, "wb") as fout:
        fout.write(struct.pack("i", GGML_MAGIC))
        fout.write(struct.pack("i", NEMOTRON_FILE_VERSION))

        for key in (
            "n_vocab",
            "n_audio_state",
            "n_audio_head",
            "n_audio_layer",
            "n_mels",
            "n_fft",
            "window_size",
            "hop_length",
            "sample_rate",
            "subsampling_factor",
            "n_subsampling_channels",
            "n_conv_kernel",
            "n_pred_dim",
            "n_pred_layers",
            "n_prompts",
            "n_max_tokens",
        ):
            fout.write(struct.pack("i", hparams[key]))

        fout.write(struct.pack("i", 1 if use_f16 else 0))

        for key in SPECIAL_TOKEN_FIELDS:
            fout.write(struct.pack("i", special_ids[key]))

        fout.write(struct.pack("i", len(tokens)))
        for token in tokens:
            write_string(fout, token)

        fout.write(struct.pack("i", len(expected_names)))
        print("\nConverting model tensors...")
        for name in expected_names:
            write_tensor(fout, name, converted_tensor_data(name, state_dict[name]), use_f16)


def convert_nemotron_to_ggml(nemo_path, output_dir, use_f16=True, out_name=None, header_path=None):
    nemo_path = Path(nemo_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if header_path is None:
        header_path = Path(__file__).resolve().parents[1] / "src" / "nemotron-arch.h"
    expected_names = parse_arch_tensor_names(header_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        safe_extract_nemo_archive(nemo_path, temp_dir)

        config_path = find_required_file(temp_dir, "model_config.yaml")
        weights_path = find_required_file(temp_dir, "model_weights.ckpt")

        config = load_model_config(config_path)
        tokenizer_model = find_nemo_resource(temp_dir, config.get("tokenizer", {}).get("model_path"))
        tokenizer_vocab = find_nemo_resource(temp_dir, config.get("tokenizer", {}).get("vocab_path"))
        if tokenizer_model is None:
            raise FileNotFoundError("SentencePiece tokenizer model from model_config.yaml was not found")
        if tokenizer_vocab is None:
            raise FileNotFoundError("Tokenizer vocab from model_config.yaml was not found")

        print(f"Found tokenizer model: {tokenizer_model.name}")
        print(f"Found tokenizer vocab: {tokenizer_vocab.name}")

        print(f"\nLoading model weights from {weights_path}")
        state_dict = load_checkpoint(weights_path)
        print(f"Loaded {len(state_dict)} tensors")

        validate_architecture_tensors(state_dict, expected_names)

        hparams = validate_hparams(config)
        tokens = load_vocabulary(config)
        special_ids = special_token_ids(tokens)

        print("\nGGML Nemotron hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")
        print("Special token ids:")
        for key in SPECIAL_TOKEN_FIELDS:
            print(f"  {key}: {special_ids[key]}")

        fname_out = output_dir / (out_name or ("ggml-model-f32.bin" if not use_f16 else "ggml-model.bin"))
        print(f"\nWriting to {fname_out}")
        write_model_file(fname_out, hparams, tokens, special_ids, state_dict, expected_names, use_f16)

    print("\nConversion complete")
    print(f"Output file: {fname_out}")
    print(f"File size: {fname_out.stat().st_size / (1024**2):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nemotron 3.5 ASR .nemo model to ggml format"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to Nemotron .nemo model file")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write ggml model file")
    parser.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
    parser.add_argument("--out-name", type=str, default=None, help="Output file name")
    parser.add_argument(
        "--arch-header",
        type=str,
        default=None,
        help="Path to src/nemotron-arch.h (default: auto-detect)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found", file=sys.stderr)
        return 1

    convert_nemotron_to_ggml(
        args.model,
        args.out_dir,
        use_f16=not args.use_f32,
        out_name=args.out_name,
        header_path=args.arch_header,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
