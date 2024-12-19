from transformers import AutoTokenizer
from datasets import Dataset
from datetime import datetime
import uuid
import json
import os
from . import logger, UserData, ExecutionTimer
from .summarizer import Summarizer

class ChatInstance:
  summarize_interval = 8  # How many new messages trigger summarization
  recent_skip = 4         # How many latest messages to skip for summarization
  max_length = 130        # Max token length before summarization

  def __init__(self, user_data: UserData):
    """Init for Chat class.

    Args:
      user_data (UserData): The data for usering.
    """
    self.chat_id = None
    self.chat_json_path = None
    self.message_history=[]
    self._summarize_index=3
    self._message_index=0
    self.summarizer=Summarizer()
    self.timer=ExecutionTimer()
    self._start(user_data=user_data)

  def _start(self, user_data: UserData) -> None:
    """Encapsulates the initialization of chat.
    Generates an ID, creates a chat log, etc.

    Args:
      user_data (UserData): The data you'd like to use.
    """
    self.chat_id = str(uuid.uuid4())[:16]

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    chat_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "chat_logs"))
    chat_filename = os.path.abspath(os.path.join(chat_dir, f"chat_{current_datetime}.json"))
    self.chat_json_path = chat_filename

    chat_data = {
      "session_info": {
        "id": self.chat_id,
        "date-time": current_datetime,
        "user_data": {
          "name": user_data.name,
          "birthday": user_data.birthday,
          "sex": user_data.sex,
          "race": user_data.race,
          "details": user_data.details
        }
      },
      "messages": []
    }

    with open(chat_filename, 'w') as json_file:
      json.dump(chat_data, json_file, indent=4)

    logger.info(f"Chat initialized. Chat data saved to {chat_filename}")

  def _append_chat_log(self, message: str, message_id: int) -> None:
    """Internal function for appending to chat log.

    Args:
      message (str): The message to append.
      message_id (int): The message ID to append.
    """    
    if self.chat_json_path is None:
      logger.error("No chat initialized. Please create a chat first.")
      return
    
    try:
      with open(self.chat_json_path, 'r+') as file:
        log_data = json.load(file)

        log_data["messages"].append({
          "message": message,
          "message_id": message_id
        })

        file.seek(0)
        json.dump(log_data, file, indent=4)
        
      logger.info(f"Message appended to chat {self.chat_json_path}")
    
    except FileNotFoundError:
      logger.error(f"File {self.chat_json_path} not found. Please initialize the chat first.")
  
  def append_message(self, role: str, message: str) -> None:
    """Function for appending messages to the message history.
    Ensures chat log writing and message index incrementing.

    Args:
      role (str): The role of the message (system, user, assistant).
      message (str): The message to append.
    """
    formatted_message = self.format_llm_text(role, message)
    self.message_history.append(formatted_message)
    self._append_chat_log(formatted_message, self._message_index)
    self._message_index += 1

  def format_llm_text(self, role: str, content: str) -> dict:
    """Easy function for formatting our LLM data.

    Args:
      role (str): The role of the message (system, user, assistant).
      content (str): The content of the message.

    Returns:
      dict: The formatted message.
    """
    return {"role": role, "content": content}
  
  def _get_summarizable_data(self) -> Dataset:
    """Internal function that batches summarizable data into a Dataset.

    Returns:
      Dataset: The summarizable data.
    """
    start_idx = self._summarize_index
    end_idx = len(self.message_history) - self.recent_skip
    logger.info(f"Processing from {start_idx} to {end_idx}.")

    summarizable_data = []
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    for i in range(start_idx, end_idx):
      tokenized_input = tokenizer(self.message_history[i]["content"], return_tensors="pt")
      token_length = tokenized_input['input_ids'].size(1)
      if token_length > self.max_length:
        summarizable_data.append({"content": self.message_history[i]["content"], "index": i})

    return Dataset.from_list(summarizable_data) if summarizable_data else None

  def _update_message_history(self, summarized_data):
    """Internal function to update the message history with the summarized messages.

    Args:
      summarized_data (_type_): The...summarized...data?
    """
    for summary in summarized_data:
      idx = summary["index"]
      self.message_history[idx]["content"] = summary["summary_text"]
      self.append_chat_log(self.message_history[idx]["content"], idx)

  def check_and_summarize(self) -> None:
    """Function to check if summary is need and perform it.
    """
    if (len(self.message_history) - self._summarize_index) >= self.summarize_interval:
      dataset = self._get_summarizable_data()
      if dataset:
        self.timer.start()
        logger.info(f"Batching data for summary: {len(dataset)} entries.")
        summarized_data = self.summarizer.summarize_batch(dataset, len(dataset))
        timer_str=self.timer.stop()
        logger.info(f"Summarizer itself took {timer_str} to finish.")
        self._update_message_history(summarized_data)
      self._summarize_index = len(self.message_history) - self.recent_skip

