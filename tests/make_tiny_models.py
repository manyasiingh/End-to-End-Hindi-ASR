"""Build tiny, randomly-initialised, fully OFFLINE models for pipeline smoke tests.
  <out>/tiny_whisper : real Whisper architecture/ID layout, d_model=64, 2+2 layers, 3-second window
  <out>/tiny_w2v2    : Wav2Vec2 (layer-norm variant, uses attention mask), hidden=64, 2 layers
These test the *code paths* (collators, labels, trainers, generate/decoding, metrics),
not accuracy on real speech."""
import json
import sys
from pathlib import Path

import torch
from transformers import (GenerationConfig, Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model,
                          WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration,
                          WhisperProcessor, WhisperTokenizer)
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode


def make_whisper(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    n_base = 50257
    byte_syms = list(bytes_to_unicode().values())
    vocab = {s: i for i, s in enumerate(byte_syms)}
    for i in range(len(vocab), n_base):
        vocab[f"<fill{i}>"] = i
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tok = WhisperTokenizer(str(out / "vocab.json"), str(out / "merges.txt"), pad_token="<|endoftext|>")
    specials = (["<|startoftranscript|>"] + [f"<|{l}|>" for l in LANGUAGES]
                + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nocaptions|>",
                   "<|notimestamps|>"] + [f"<|{i * 0.02:.2f}|>" for i in range(1501)])
    tok.add_tokens(specials, special_tokens=True)
    tok.add_special_tokens({"additional_special_tokens": specials})
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    print("eot", eot, "sot", sot, "hi", tok.convert_tokens_to_ids("<|hi|>"), "len", len(tok))
    vocab_size = len(tok)

    cfg = WhisperConfig(vocab_size=vocab_size, d_model=64, encoder_layers=2, decoder_layers=2,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=128,
                        decoder_ffn_dim=128, num_mel_bins=80, max_source_positions=150,
                        max_target_positions=448, pad_token_id=eot, bos_token_id=eot, eos_token_id=eot,
                        decoder_start_token_id=sot, suppress_tokens=[], begin_suppress_tokens=[],
                        forced_decoder_ids=None)
    model = WhisperForConditionalGeneration(cfg)
    lang_to_id = {f"<|{l}|>": tok.convert_tokens_to_ids(f"<|{l}|>") for l in LANGUAGES}
    model.generation_config = GenerationConfig(
        decoder_start_token_id=sot, eos_token_id=eot, pad_token_id=eot, bos_token_id=eot,
        max_length=448, is_multilingual=True, lang_to_id=lang_to_id,
        task_to_id={"translate": tok.convert_tokens_to_ids("<|translate|>"),
                    "transcribe": tok.convert_tokens_to_ids("<|transcribe|>")},
        no_timestamps_token_id=tok.convert_tokens_to_ids("<|notimestamps|>"),
        prev_sot_token_id=tok.convert_tokens_to_ids("<|startofprev|>"),
        suppress_tokens=[], begin_suppress_tokens=[], return_timestamps=False)
    model.save_pretrained(out)
    fe = WhisperFeatureExtractor(feature_size=80, chunk_length=3)  # 3-second window => 300 frames
    WhisperProcessor(feature_extractor=fe, tokenizer=tok).save_pretrained(out)
    print("tiny whisper params:", sum(p.numel() for p in model.parameters()))


def make_w2v2(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cfg = Wav2Vec2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                         conv_dim=(32,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
                         num_feat_extract_layers=7, feat_extract_norm="layer", do_stable_layer_norm=True,
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
                         mask_time_prob=0.0, vocab_size=32)
    Wav2Vec2Model(cfg).save_pretrained(out)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=True).save_pretrained(out)
    print("tiny w2v2 saved")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke/models")
    torch.manual_seed(0)
    make_whisper(root / "tiny_whisper")
    make_w2v2(root / "tiny_w2v2")
