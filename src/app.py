from llmimic import LLMInstance, UserData

if __name__ == "__main__":
  llm_instance=LLMInstance()
  user_data = UserData("John Doe", "1970-01-01", "Male", "Caucasian", None)
  llm_instance.start(user_data, "default")
  response=llm_instance.generate_response("Who was the 10th president of the United States of America?")
  print(response)