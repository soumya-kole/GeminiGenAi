import re

from Levenshtein import ratio
import json


def preprocess_text(text):
    """Lowercase, remove extra spaces and special characters for better comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r"[^\w\s]", '', text) # Remove punctuation
    return text


def calculate_ocr_accuracy(ocr_text, json_text):
    """Computes character-level and word-level accuracy using similarity metrics."""
    ocr_text = preprocess_text(ocr_text)
    json_text = preprocess_text(json_text)

    # Character-level accuracy using Levenshtein ratio
    char_accuracy = ratio(ocr_text, json_text) * 100

    # Word-level accuracy
    ocr_words = set(ocr_text.split())
    json_words = set(json_text.split())

    common_words = ocr_words.intersection(json_words)
    word_accuracy = (len(common_words) / len(json_words)) * 100 if json_words else 0
    # missing_words = json_words - ocr_words
    # print("Missing Words:", missing_words)
    # extra_words = ocr_words - json_words
    # print("Extra Words:", extra_words)
    return {
        "Character-Level Accuracy": f"{char_accuracy:.2f}%",
        "Word-Level Accuracy": f"{word_accuracy:.2f}%",
    }



def parse_funsd_json(json_path):
    """Parse FUNSD JSON file to extract structured text while maintaining question-answer relationships."""
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    structured_text = []
    entity_dict = {}

    # Step 1: Organize entities into a dictionary
    for item in json_data["form"]:
        entity_dict[item["id"]] = {
            "text": item["text"].strip(),
            "label": item["label"],
            "linked_to": item["linking"]
        }

    # Step 2: Reconstruct meaningful text preserving structure
    for entity_id, entity in entity_dict.items():
        text = entity["text"]
        label = entity["label"]

        # If it's a header, add it as a separate section
        if label == "header":
            structured_text.append(f"{text.upper()}")

        # If it's a question, format it properly
        elif label == "question":
            structured_text.append(f"{text}")

            # Check if it's linked to an answer
            linked_answers = [
                entity_dict[linked_id[1]]["text"]
                for linked_id in entity["linked_to"]
                if linked_id[1] in entity_dict and entity_dict[linked_id[1]]["label"] == "answer"
            ]

            if linked_answers:
                structured_text.append(f"{', '.join(linked_answers)}")

        # If it's an independent answer, list it separately
        elif label == "answer" and not entity["linked_to"]:
            structured_text.append(f"{text}")

    # Join the formatted text
    return "\n".join(structured_text)


def main(extracted_file, json_path):
    with open(extracted_file, 'r') as file:
        ocr_text = file.read()

    accuracy_results = calculate_ocr_accuracy(ocr_text, parse_funsd_json(json_path))
    print("OCR Accuracy Results:", accuracy_results)

if __name__ == '__main__':
    main('../extracted_text/82092117_Gemini.txt', '82092117.json')
    main('../extracted_text/82092117_OpenAI.txt', '82092117.json')
    main('../extracted_text/82200067_0069_Gemini.txt', '82092117.json')
    main('../extracted_text/82200067_0069_OpenAI.txt', '82092117.json')
    main('../extracted_text/82251504_Gemini.txt', '82251504.json')
    main('../extracted_text/82251504_OpenAI.txt', '82251504.json')
