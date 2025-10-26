#!/usr/bin/env python3
"""
Download wav2vec2 audio processor config files.

This script downloads the necessary configuration files for the wav2vec2
audio processor used in Wan2.2-S2V-14B training. These are small JSON files
(~4KB total) that can be committed to the repository.

The actual model weights (model.safetensors, 1.2GB) should be downloaded
separately via ModelScope and stored on /ssd2.

Usage:
    python scripts/download_wav2vec_configs.py
"""

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Config
import os

# Target directory (in codebase, small config files can be committed)
save_path = 'models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english'
os.makedirs(save_path, exist_ok=True)

print("=" * 60)
print("Downloading wav2vec2 audio processor config files")
print("=" * 60)

# Step 1: Download feature extractor config (preprocessor_config.json)
print("\n[1/3] Downloading feature extractor config...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
feature_extractor.save_pretrained(save_path)
print(f"✅ Feature extractor saved: preprocessor_config.json")

# Step 2: Download model config (config.json)
print("\n[2/3] Downloading model config...")
config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-large-xlsr-53')
config.save_pretrained(save_path)
print(f"✅ Model config saved: config.json")

# Step 3: Download tokenizer configs (tokenizer_config.json, special_tokens_map.json, vocab.json)
print("\n[3/3] Downloading tokenizer configs...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
tokenizer.save_pretrained(save_path)
print(f"✅ Tokenizer configs saved: tokenizer_config.json, special_tokens_map.json, vocab.json")

# Verify all files are present
print("\n" + "=" * 60)
print("Verification:")
print("=" * 60)
required_files = [
    'preprocessor_config.json',
    'config.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.json'
]

all_present = True
total_size = 0
for file in required_files:
    path = os.path.join(save_path, file)
    if os.path.exists(path):
        size = os.path.getsize(path)
        total_size += size
        print(f"✅ {file:30s} ({size:,} bytes)")
    else:
        print(f"❌ {file:30s} MISSING")
        all_present = False

print("-" * 60)
print(f"Total size: {total_size:,} bytes (~{total_size/1024:.1f} KB)")
print("=" * 60)

if all_present:
    print("\n🎉 SUCCESS! All config files downloaded.")
    print(f"\nLocation: {os.path.abspath(save_path)}/")
    print("\nThese small config files can be committed to git.")
    print("The large model weights (model.safetensors, 1.2GB) should be")
    print("downloaded separately via ModelScope and stored on /ssd2.")
else:
    print("\n❌ ERROR: Some files are missing!")
    exit(1)
