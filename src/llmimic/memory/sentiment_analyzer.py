import torch
from transformers import pipeline
from datasets import Dataset
import os
import json
import nltk
nltk.download('punkt_tab')

class SentimentAnalyzer:  
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=device
        )
    
    def analyze_sentiment(self, text: str, batch_size=8):
        """Analyzes the string for sentiment.

        Args:
            text (str): The text to analyze.
            batch_size (int, optional): The batch size to analyze. Defaults to 8.

        Returns:
            _type_: The sentiment data.
        """
        sentences = nltk.sent_tokenize(text)
        
        dataset = Dataset.from_dict({"text": sentences})
        
        def process_batch(batch):
            print(batch)
            results = self.sentiment_analyzer(batch["text"], batch_size=batch_size)
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_2': 'positive'
            }
            
            sentiments = []
            for i, res in enumerate(results):
                label = label_mapping.get(res['label'], 'unknown')
                score = res['score']
                sentence = batch["text"][i]
                
                if res['label'] != 'LABEL_1':
                    sentiments.append({
                        "label": label,
                        "score": score,
                        "sentence": sentence
                    })
                else:
                    sentiments.append(None)
                
            return {"sentiments": sentiments}

        processed_dataset = dataset.map(process_batch, batched=True)
        
        return [item for item in processed_dataset["sentiments"] if item is not None]
    
    def append_to_sentiment_json(self, role: str, memory_data, memory_dir: str) -> None:
        """Appends the data to the JSON file.

        Args:
            role (str): The role of the message.
            memory_data (_type_): The sentiment data.
            memory_dir (str): The memory directory.
        """
        sentiment_file = os.path.join(memory_dir, 'sentiment.json')

        if not os.path.isfile(sentiment_file):
            with open(sentiment_file, 'w') as file:
                json.dump([], file)

        with open(sentiment_file, 'r') as file:
            data = json.load(file)

        for entry in memory_data:
            entry_data = {
                "memory_data": {
                    "role": role,
                    "sentiment": entry["label"],
                    "score": entry["score"],
                    "sentence": entry["sentence"]
                }
            }

            data.append(entry_data)

        with open(sentiment_file, 'w') as file:
            json.dump(data, file, indent=4)