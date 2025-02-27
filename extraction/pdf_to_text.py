import os
import logging
from pdf2image import convert_from_path
import google.genai as genai
from google.genai.types import Part, Image
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.getenv("PROJECT"),
    location=os.getenv("LOCATION"),
)


def get_text_from_image(image_path):
    model = "gemini-2.0-flash-001"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        response = client.models.generate_content(
            model=model,
            contents=[
                "Extract text from the image.",  # Wrap the instruction in Part.from_text()
                Part.from_bytes(data=image_data, mime_type="image/jpeg"),  # Send image as input
            ],
        )
    return response.text


def pdf_to_images(pdf_file_with_path, output_folder="output_images", dpi=300, fmt="jpeg"):
    pdf_name = os.path.basename(pdf_file_with_path).replace(".pdf", "")
    output_folder = os.path.join(output_folder, pdf_name)
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_file_with_path, dpi=dpi)

    image_paths = []
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.{fmt}"
        image.save(image_path, fmt.upper())
        image_paths.append(image_path)

    logger.info(f"{len(image_paths)} images saved in '{output_folder}'")
    return image_paths


def pdf_to_text(pdf_path, text_folder="output_texts"):
    for f in os.listdir(pdf_path):
        if f.endswith(".pdf"):
            pdf_file_path = os.path.join(pdf_path, f)
            output_file = os.path.join(text_folder, f.replace(".pdf", ".txt"))
            image_files = pdf_to_images(pdf_file_path)
            extracted_texts = []
            for image_file in image_files:
                extracted_text = get_text_from_image(image_file)
                extracted_texts.append(extracted_text)
            text_folder = os.path.dirname(output_file)
            os.makedirs(text_folder, exist_ok=True)
            with open(output_file, "w") as text_file:
                logger.info(f"Extracted text from {f} and saved to {output_file}")
                for text in extracted_texts:
                    text_file.write(text + '\n')


if __name__ == "__main__":
    pdf_to_text("pdfs", "output_texts")
