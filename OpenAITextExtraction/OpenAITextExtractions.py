import base64
import os

import openai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from tqdm import tqdm

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")

# Directory paths
PDF_DIR = "../pdf_files"  # Folder containing PDFs
OUTPUT_DIR = "../extracted_text"  # Folder to store extracted text

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pdf_to_images(pdf_path):
    """Convert a PDF into a list of images (one per page)."""
    images = convert_from_path(pdf_path)
    return images


def image_to_text(image):
    """Extract text from an image using OpenAI GPT-4 Vision."""
    # Convert image to Base64
    with open("temp.jpg", "wb") as temp_file:
        image.save(temp_file, format="JPEG")

    with open("temp.jpg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Call GPT-4 Vision using the OpenAI package
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert OCR assistant. Extract all readable text from this document image accurately."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract all text from the document and retain the original formatting as closely as possible. Pay close attention to tables, lists, and headers."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=1000
    )

    return response["choices"][0]["message"]["content"]


def process_pdf(pdf_path):
    """Convert a PDF to images and extract text from each page."""
    print(f"Processing: {pdf_path}")
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_OpenAI.txt"))
    if os.path.exists(output_file):
        print(f"Text file '{output_file}' already exists. Skipping...")
        return
    images = pdf_to_images(pdf_path)
    extracted_text = []

    for idx, image in enumerate(tqdm(images, desc="Extracting text from pages")):
        text = image_to_text(image)
        extracted_text.append(text)

    # Save extracted text

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(extracted_text)

    print(f"Text extracted and saved to {output_file}")


# Process all PDFs in the directory
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the directory.")
    else:
        for pdf_file in pdf_files:
            process_pdf(os.path.join(PDF_DIR, pdf_file))
