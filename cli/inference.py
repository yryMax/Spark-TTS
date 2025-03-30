# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform

from cli.SparkTTS import SparkTTS

model = SparkTTS("pretrained_models/Spark-TTS-0.5B", 'cuda:0')

def synthesize(args):

    # Initialize the model


    save_path = args.output_dir

    logging.info("Starting inference...")

    with torch.no_grad():
        wav = model.inference(
            args.text,
            args.ref_audio,
            prompt_text=args.prompt_text,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPARK TTS CLI")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio file")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt_text", type=str, required=True, help="Prompt text")

    args = parser.parse_args()

    synthesize(args)
