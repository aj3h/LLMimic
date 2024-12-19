from . import logger, UserData, ExecutionTimer
from .chat_instance import ChatInstance
import json
import os
import torch
from transformers import pipeline
from datetime import datetime
import requests
import re
from .memory import MemoryInstance

class LLMInstance:
  """The main LLM instance class: calls chat_instance, memory_instance, chess_instance, etc.
  """
  def __init__(self):
    """Ensure configs in /configs/ are set properly before instancing.
    """
    self.pipe=None
    self.llm_active=False
    self.model_name = None
    self.persona_name = None
    self.preset = None
    self.weather_api_key = None
    self.lat = None
    self.lon = None
    self.use_memory=False
    self.get_weather=False
    self.memory_instance=None
    self.chat_instance = None
    self.load_config()
    logger.info("An instance has been initialized.")

  def load_config(self) -> None:
    """Attempts to load defaults from configs/config.json.

    Raises:
        ValueError: Latitude or longitude is missing in config.
        ValueError: There was an error parsing the config file.
        FileNotFoundError: The file needs to be configs/config.json.
    """
    config_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))
    config_file=os.path.abspath(os.path.join(config_dir, "config.json"))
    if os.path.exists(config_file):
      try:
        with open(config_file, "r") as file:
          data = json.load(file)
          self.model_name = data.get("llama_small_model")
          self.persona_name = data.get("persona_name", "generic")
          self.preset = data.get("preset", "default")
          self.weather_api_key = data.get("weather_api_key", "YOUR_API_KEY_HERE")
          self.lat = data.get("weather_lat")
          self.lon = data.get("weather_lon")
          self.use_memory = data.get("use_memory")
          self.get_weather = data.get("get_weather")

          if self.lat is None or self.lon is None:
            raise ValueError("Latitude or Longitude is missing in the configuration.")
      except json.JSONDecodeError as e:
        error_message = f"Error parsing {config_file}: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    else:
      error_message = f"{config_file} not found. Halting execution."
      logger.error(error_message)
      raise FileNotFoundError(error_message)
    
  def is_active(self) -> bool:
    """A completely trivial and unnecessary is active function.

    Returns:
        bool: True if there's a session or chat isn't None.
    """
    if self.llm_active or self.chat_instance is not None:
      return True
    else:
      return False
    
  def start(self, user_data: UserData, preset_name: str) -> bool:
    """Function to start a chat/session.

    Args:
        user_data (UserData): UserData type containing...well, user data.
        preset_name (str): The preset from configs/presets.json for session.

    Returns:
        bool: Whether successful, but it's never actually checked.
    """
    if self.is_active():
      logger.error("A session or chat is already active.")
      return False
    timer=ExecutionTimer()
    timer.start()
    llm_persona_info=self._construct_persona_info()
    if self.get_preset_data(preset_name):
      LLM_SYS=llm_persona_info + " " + self.system_message
      self.pipe = pipeline(
        "text-generation",
        model=self.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
      )
    else:
      logger.error("Failed to load preset data.")
      return False
    USER_INFO=self._construct_user_data(user_data=user_data)
    LLM_INTRO="Okay, got it! I'll remember that for reference later! Let's get started! I'll wait for you to greet me."
    self.chat_instance=ChatInstance(user_data=user_data)
    logger.info("User info and LLM intro created successfully.")
    
    self.chat_instance.append_message("system", LLM_SYS)
    self.chat_instance.append_message("user", USER_INFO)
    self.chat_instance.append_message("assistant", LLM_INTRO)
    
    if self.use_memory:
      self.memory_instance=MemoryInstance(os.path.abspath(os.path.join(os.path.dirname(__file__), f"persona/{self.persona_name}")))

    self.llm_active=True
    timer_str=timer.stop()
    logger.info(f"Chat logs, initial messages, and LLM session active, total time: {timer_str}.")
    logger.info(f"Session started for a {user_data.race} {user_data.sex} named {user_data.name}, born {user_data.birthday}.")
    return True

  def _process_dict(self, data, path="") -> str:
    """Internal function to process the persona data into a string.

    Args:
        data (JSON): The JSON data retrieved from the persona.json file.
        path (str, optional): Genuinely don't remember what this is for. Defaults to "".

    Returns:
        str: The persona data as a chunk of text the LLM enjoys.
    """
    result = []
    for key, value in data.items():
        current_path = f"{path}({key})" if path else key
        if isinstance(value, dict):
          nested_result = self._process_dict(value, path=current_path)
          result.append(nested_result)
        elif isinstance(value, list):
          list_values = ", ".join(map(str, value)) if value else "None"
          result.append(f"{current_path}: {list_values}")
        else:
          result.append(f"{current_path}: {value}")
    return ", ".join(result)

  def _construct_persona_info(self) -> str:
    """This is the main function for constructing the persona data.
    It grabs the JSON data, calls _process_dict(), appends prefix/suffix, etc.

    Returns:
      str: The complete persona string that gets fed to the LLM.
    """
    logger.info("Constructing persona data.")
    timer=ExecutionTimer()
    timer.start()
    persona_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), f"persona/{self.persona_name}"))
    persona_json=os.path.abspath(os.path.join(persona_dir, f"{self.persona_name}.json"))
    try:
      with open(persona_json, 'r') as file:
        json_data = json.load(file)

      persona_info = self._process_dict(json_data)

      prefix = "Persona Information: "
      suffix = " End of persona data."
      timer_str = timer.stop()
      logger.info(f"Constructed persona data in {timer_str}.")
      return f"{prefix}{persona_info}{suffix}"

    except FileNotFoundError:
      logger.error(f"File '{persona_json}' not found.")
      timer_str = timer.stop()
      logger.info(f"Failed to construct persona data in {timer_str}.")
      return None
    except json.JSONDecodeError:
      logger.error(f"File '{persona_json}' is not a valid JSON file.")
      timer_str = timer.stop()
      logger.info(f"Failed to construct persona data in {timer_str}.")
      return None
    
  def get_preset_data(self, preset_name='default') -> bool:
    """Function to get the preset data from configs/presets.json.
    Ultimately responsible for loading generation parameters.

    Args:
      preset_name (str, optional): The name of the preset to load. Defaults to 'default'.

    Raises:
      KeyError: The presets key wasn't found in JSON file.
      KeyError: The preset wasn't found in the JSON file.

    Returns:
      bool: Whether successful or not.
    """
    config_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))
    preset_json=os.path.abspath(os.path.join(config_dir, "presets.json"))
    try:
      with open(preset_json, 'r') as file:
        data = json.load(file)

      if "presets" not in data:
        raise KeyError("'presets' key not found in JSON file.")

      presets = data["presets"]

      if preset_name not in presets:
        raise KeyError(f"Preset '{preset_name}' not found in JSON file.")

      config = presets[preset_name]

      setattr(self, "system_message", config.get("system_message", ""))
      setattr(self, "max_tokens", config.get("max_tokens", 0))
      setattr(self, "temperature", config.get("temperature", 0.0))
      setattr(self, "top_p", config.get("top_p", 0.0))
      setattr(self, "top_k", config.get("top_k", 0))
      setattr(self, "repetition_penalty", config.get("repetition_penalty", 0.0))
      setattr(self, "num_beams", config.get("num_beams", 1))
      setattr(self, "length_penalty", config.get("length_penalty", 0.0))
      return True

    except FileNotFoundError:
      logger.error(f"File '{preset_json}' not found.")
      return False
    except json.JSONDecodeError:
      logger.error(f"File '{preset_json}' is not a valid JSON file.")
      return False
    except KeyError as e:
      logger.error(f"{e}")
      return False
    
  def _construct_user_data(self, user_data: UserData) -> str:
    """Function to construct LLM text chunks for UserData.
    If get_weather is enabled, this is where it's called.

    Args:
      user_data (UserData): The umm...data from the user.

    Returns:
      str: The complete string of user data to feed the LLM.
    """
    logger.info("Constructing user data.")
    timer=ExecutionTimer()
    timer.start()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if self.get_weather:
      weather_info = self._get_weather_info(self.weather_api_key, self.lat, self.lon)

      user_data = {
          "name": user_data.name,
          "race": user_data.race,
          "sex": user_data.sex,
          "birthday": user_data.birthday,
          "details": user_data.details,
          "current_time": current_time,
          "weather": {
            "city": weather_info['city'],
            "country": weather_info['country'],
            "weather_main": weather_info['weather_main'],
            "description": weather_info['description'],
            "temperature": weather_info['temperature'],
            "feels_like": weather_info['feels_like']
          }
      }
    else:
      user_data = {
      "name": user_data.name,
      "race": user_data.race,
      "sex": user_data.sex,
      "birthday": user_data.birthday,
      "details": user_data.details,
      "current_time": current_time
      }

    prefix = "User Information: "
    suffix = "End of user information."

    user_data_message = prefix + str(user_data) + suffix
    timer_str = timer.stop()
    logger.info(f"Constructed user data in {timer_str}.")
    return user_data_message

  def _get_weather_info(self, api_key: str, lat: float, lon: float) -> dict:
    """Function to get the current weather at a location.
    Utilizes OpenWeatherAPI and requires an API key, a latitude, and a longitude.

    Args:
      api_key (str): Your OpenWeatherAPI API key.
      lat (float): The latitude to get the weather data from.
      lon (float): The longitude to get the weather data from.

    Returns:
      dict: The weather data for the chosen location (on success, N/A otherwise).
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
    "lat": lat,
    "lon": lon,
    "appid": api_key,
    "units": "imperial"
    }

    default_result = {
    "weather_main": "N/A",
    "description": "N/A",
    "temperature": "N/A",
    "feels_like": "N/A",
    "wind_speed": "N/A",
    "city": "N/A",
    "country": "N/A"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "weather" not in data or "main" not in data or "sys" not in data:
          return default_result

        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})

        result = {
        "weather_main": weather.get("main", "N/A").lower(),
        "description": weather.get("description", "N/A").lower(),
        "temperature": main.get("temp", "N/A"),
        "feels_like": main.get("feels_like", "N/A"),
        "wind_speed": wind.get("speed", "N/A"),
        "city": data.get("name", "N/A"),
        "country": data.get("sys", {}).get("country", "N/A")
        }
        return result
    except requests.RequestException as e:
        return default_result
    
  def trim_after_last_punctuation(self, text: str) -> str:
    """Trims the trailing end off of LLM responses (hung sentences).

    Args:
      text (str): The text to trim.

    Returns:
      str: The trimmed text. Genius!
    """
    matches = re.findall(r'[.?!](?=\s|$)', text)
    
    if matches:
      last_punct_index = text.rfind(matches[-1])
      return text[:last_punct_index + 1]
    return text
    
  def generate_response(self, prompt: str) -> str:
    """The bread and butter of response generation.
    This is also where summarization and memory happen.

    Args:
      prompt (str): The prompt to submit (last prompt).

    Returns:
      str: The generated response, probably.
    """
    logger.info("Submitting prompt for generation.")
    timer=ExecutionTimer()
    timer.start()
    self.chat_instance.append_message("user", prompt)
    self.chat_instance.check_and_summarize()
    self.memory_instance.check_for_memories("user", prompt)
    outputs = self.pipe(
      self.chat_instance.message_history,
      max_new_tokens=self.max_tokens,
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      repetition_penalty=self.repetition_penalty,
      num_beams=self.num_beams,
      length_penalty=self.length_penalty,
    )
    
    response = outputs[0]["generated_text"][-1]
    trimmed_response = self.trim_after_last_punctuation(response['content'])
    self.chat_instance.append_message("assistant", trimmed_response)
    self.chat_instance.check_and_summarize()
    self.memory_instance.check_for_memories("assistant", trimmed_response)
    timer_str=timer.stop()
    logger.info(f"Prompt response generated, total time: {timer_str}.")
    return trimmed_response