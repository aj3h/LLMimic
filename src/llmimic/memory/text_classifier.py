import torch
from transformers import pipeline
import os
import json
from datetime import datetime
from llmimic import logger

class TextClassifier:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
        self.__generate_candidate_labels()

    def __generate_candidate_labels(self):
        """Internal function generation candidate labels for categorization.
        Attempted to place emphasis on potentially emotionally relevant moments.
        """
        throwaway_labels = [
            "news", "general conversation", "weather", "sports", "technology", "politics", "entertainment",
            "travel", "work", "shopping", "food", "social media"
        ]
        
        personal_labels = [
            "first online conversation", "first voice call", "first impression", "first surprise",
            "emotional vulnerability", "deepening trust", "early jealousy", "expressing desires",
            "growing together as a team", "making compromises", "creating traditions together", "celebrating small wins together",
            "expressing gratitude", "first time meeting family", "first holiday together", "first joint decision", "first gift exchange",
            "first travel experience together", "first shared hobby or passion", "first time spending the night together",
            "moving in together milestone", "buying a home together", "moment of emotional support", "comfort during tough times",
            "physical closeness", "heartfelt confession", "learning each other's love language", "growing intimacy",
            "navigating personal growth together", "being each other's safe space", "first time saying 'I love you'",
            "first intimate moment", "shared dreams of the future", "sensual connection", "first disagreement", "dealing with insecurities",
            "making up after a fight", "emotional repair", "making the first apology", "acknowledging faults and flaws",
            "finding compromise after conflict", "working through misunderstandings", "planning future together", "milestone anniversaries",
            "reaffirming commitment", "loss of a loved one together", "overcoming a betrayal", "first long distance separation",
            "emotional crisis", "spontaneous adventure", "inside joke", "cute gesture", "shared hobbies", "silly inside jokes",
            "spontaneous acts of kindness", "playful teasing", "surprising each other", "support during career change", "first time living together",
            "parenting moment", "pet adoption together", "adopting a shared responsibility", "joint problem-solving", "balancing independence and togetherness",
            "celebrating small victories", "random acts of romance", "seasonal traditions"
        ]

        self.personal_labels = personal_labels
        self.throwaway_labels = throwaway_labels

    def classify_text(self, text: str) -> str:
        """The function that classifies the text.

        Args:
            text (str): The text to classify.

        Returns:
            str: The classification it found.
        """
        result = self.classifier(text, self.personal_labels + self.throwaway_labels)
        return result['labels'][0]
    
    def append_to_memory(self, classification, role: str, original_text: str, memory_dir: str):
        """Appends the memory data to the JSON file.

        Args:
            classification (_type_): The classification data.
            role (str): The role of the message.
            original_text (str): The original message content.
            memory_dir (str): The memory directory.
        """
        if classification not in self.personal_labels:
            logger.info(f"Ignored classification: {classification}")
            return

        memory_file = os.path.join(memory_dir, 'classification.json')

        memory_data = {
            "datetime": datetime.now().isoformat(),
            "classification": classification,
            "role": role,
            "message": original_text
        }

        if not os.path.exists(memory_file):
            with open(memory_file, 'w') as file:
                json.dump({"memory_data": [memory_data]}, file, indent=4)
        else:
            with open(memory_file, 'r+') as file:
                data = json.load(file)
                data["memory_data"].append(memory_data)
                file.seek(0)
                json.dump(data, file, indent=4)