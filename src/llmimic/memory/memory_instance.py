from .entity_recognizer import EntityRecognizer
from .sentiment_analyzer import SentimentAnalyzer
from .text_classifier import TextClassifier
import os
from llmimic import logger, ExecutionTimer

class MemoryInstance:
  def __init__(self, memory_dir_path):
    self.memory_dir=os.path.abspath(os.path.join(memory_dir_path, "memory_data"))
    self.entity_recognizer=None
    self.sentiment_analyzer=None
    self.text_classifier=None
    self.timer=ExecutionTimer()
    logger.info("A memory instance has been initialized.")

  def check_for_memories(self, role: str, text: str):
    """This is a simple and horrible implementation currently.
    None of this stuff is properly implemented in any way.
    We're just working with data currently for future refactoring.

    Args:
        role (str): The role of the text being checked.
        text (str): The text to be checked.
    """    
    logger.info("Checking for memories.")
    self.timer.start()
    self.entity_recognizer=EntityRecognizer()
    entity_list=self.entity_recognizer.analyze_entities(text)
    self.entity_recognizer.process_entity_memory(entity_list, role, text, self.memory_dir)
    del self.entity_recognizer
    self.entity_recognizer=None
    self.sentiment_analyzer=SentimentAnalyzer()
    sentiment_data=self.sentiment_analyzer.analyze_sentiment(text)
    self.sentiment_analyzer.append_to_sentiment_json(role, sentiment_data, self.memory_dir)
    del self.sentiment_analyzer
    self.sentiment_analyzer=None
    self.text_classifier=TextClassifier()
    classification=self.text_classifier.classify_text(text)
    self.text_classifier.append_to_memory(classification, role, text, self.memory_dir)
    del self.text_classifier
    self.text_classifier=None
    timer_str=self.timer.stop()
    logger.info(f"Memory processes finished in {timer_str}.")