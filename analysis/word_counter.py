import os
import re
import pandas as pd
from collections import Counter
def word_count(text):
  clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  words = clean_text.split()
  return len(words)

def word_count_dataframe(text):
    """
    Processes the given text, removes non-alphanumeric characters,
    counts individual word occurrences, and returns a Pandas DataFrame.
    """
    # Remove non-alphanumeric characters except spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Split text into words
    words = cleaned_text.lower().split()

    # Count word occurrences
    word_count = Counter(words)

    # Convert to Pandas DataFrame
    df_word_count = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)

    return df_word_count
if __name__ == "__main__":
  # Read the text from a file
  text_folder ='/Users/soumya/Technicals/pythonProject/GeminiGenAI/extraction/output_texts/'
  for text_file in os.listdir(text_folder):
      with open(os.path.join(text_folder, text_file), 'r') as file:
        text = file.read()
        count = word_count(text)
        df = word_count_dataframe(text)
        print(text_file, " => Word count:", count)
        print(df)
