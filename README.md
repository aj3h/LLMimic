# LLMimic

## Disclaimer

This repository is a **personal project**, created solely for my own experimentation and as a proof of concept. It is not intended for public use or consumption.

### Important Notes:
1. **No Guarantees:**
- The content in this repository is provided *as-is*.
- There are no guarantees of functionality, correctness, or suitability for any purpose.

2. **No Liability:**
- I accept no responsibility for any issues, errors, or damages resulting from the use of this code or its components.

3. **No Support or Collaboration:**
- I am not accepting issues, pull requests, or feature requests.
- This project is not open for collaboration, feedback, or contributions.

By accessing this repository, you acknowledge and accept these terms.

## Overview
LLMimic is a modular Python framework designed to personalize interactions with LLaMa-based LLMs. It supports managing multiple customizable personas, a WIP memory system, and optional features like weather integration and chess gameplay. Perfect for exploring AI-human interaction with a personal touch.

## Features
- **Persona Management**: Create and manage multiple personas with rich backstories.
- **Memory System (WIP)**: Entity recognition, sentiment analysis, and text classification with JSON storage.
- **Chat Summarizer**: Reduces token usage for longer conversations.
- **Weather Integration**: Fetch real-time geographical weather with [OpenWeatherMap](https://openweathermap.org/current).
- **Toggleable Modules/Features**: Enable or disable modules and features for flexibility.
- **Chess Module**: Play chess alongside chatting.

## Current Limitations
- Memory data is stored but not integrated into responses.
- No token management for very long chats.
- Chess LLM is not directly integrated with the Persona LLM.
- Planned features (fine-tuning, randomness module) are in early stages.

## Project Setup

### Install CUDA Toolkit 12.4

- Download and install the [Nvidia CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) from the official site.
- Install the [Nvidia CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) by following the instructions on the official website.

### Create the Conda environment

```bash
$ conda create -p .venv python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
$ conda install transformers accelerate sentencepiece protobuf chess nltk
```

## Configuration JSONs

There are three JSON files responsible for various aspects of LLMimic:
- llmimic/configs/config.json: this where you set models, presets, persona, [OpenWeatherMap](https://openweathermap.org/current) API data, and other app configs.
- llmimic/configs/presets.json: this is where the preset data is stored (system message and generation params).
- llmimic/persona/<persona_name>/<persona_name>.json: files that define attributes of the persona.

### config.json

This contains the general app/module configuration data. You can change the current LLaMa models by selecting a new one from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads). Changing the model itself hasn't been easily implemented though and requires altering this code within llm_instance.py:

```python
self.model_name = data.get("llama_small_model")
```

### presets.json

The presets are fixed, as the current structure is expected. You can alter the contents or add new presets to suit your needs though. All the fields should be self-explanatory if you've ever worked with LLMs before. Just ensure the format is followed.

### persona.json

Currently, persona are self-contained and highly customizable. To create a new persona, simply copy/paste the 'generic' directory and rename both the 'generic' directory and the 'generic.json' to the same name. The contents of the persona JSON are mostly freeform. It is recommended to keep the 'core identity', but pretty much anything can be removed, added, or altered (as far as testing has shown).

## Usage

The default application file (app.py) shows a general example for console use:

```python
llm_instance=LLMInstance() #Create a new instance.
user_data = UserData("John Doe", "1970-01-01", "Male", "Caucasian", None) #Submit the user data.
llm_instance.start(user_data, "default") #Start the session.
response=llm_instance.generate_response("Who was the 10th president of the United States of America?")
print(response) #Prompt, generate, and print.
```