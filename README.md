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
