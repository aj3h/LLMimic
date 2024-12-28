import os
import json
from nltk.tokenize import sent_tokenize
import torch
from transformers import pipeline
from datetime import datetime

class EntityRecognizer:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_model = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=device
        )
        
    def analyze_entities(self, text: str):
        """Analyzes the text for recognizable entities.

        Args:
            text (str): The text to analyze.

        Returns:
            list?: The entities it found.
        """
        ner_results = self.ner_model(text)

        entities = []
        current_entity = None
        current_label = None

        for result in ner_results:
            word = result['word']
            label = result['entity_group']

            if word.startswith('##'):
                if current_entity:
                    current_entity += word[2:]
            else:
                if current_entity and current_label:
                    entities.append((current_entity, current_label))

                current_entity = word
                current_label = label

        if current_entity and current_label:
            entities.append((current_entity, current_label))

        return entities
    
    def process_entity_memory(self, entity_list, role: str, text: str, memory_dir: str):
        """Processes the entity data into JSON data for our memory.

        Args:
            entity_list (list?): The entities we found with analyze_entities.
            role (str): The role of the message.
            text (str): The message data itself.
            memory_dir (str): The memory directory.
        """
        sentences = sent_tokenize(text)

        os.makedirs(memory_dir, exist_ok=True)

        file_path = os.path.join(memory_dir, 'entity_recognition.json')

        new_entries = []

        for entity, label in entity_list:
            matching_sentences = [
                sentence for sentence in sentences if entity.lower() in sentence.lower()
            ]
            combined_sentences = " ".join(matching_sentences)

            if matching_sentences:
                new_entries.append({
                    "date-time": datetime.now().isoformat(),
                    "role": role,
                    "entity": entity,
                    "label": label,
                    "sentences": combined_sentences
                })

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        existing_data.extend(new_entries)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, indent=4)
