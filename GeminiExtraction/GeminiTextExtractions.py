import os
from google.cloud import storage
import google.genai as genai
from dotenv import load_dotenv
from google.genai.types import Part

# Load environment variables
load_dotenv()

# Google Cloud Configuration
BUCKET_NAME = "genaitestsk"
PDF_DIR = "../pdf_files"  # Folder containing PDFs
OUTPUT_DIR = "../extracted_text"  # Folder to save extracted text

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Google Cloud Storage client
storage_client = storage.Client(os.getenv("PROJECT"))

# Initialize Gemini model
model = "gemini-2.0-flash-001"
client = genai.Client(
    vertexai=True,
    project=os.getenv("PROJECT"),
    location=os.getenv("LOCATION"),
)

# System and User Prompts
SYSTEM_PROMPT = """You are an expert OCR assistant. Extract all readable text from this document accurately."""
USER_PROMPT = """Extract all text from the document and retain the original formatting as closely as possible. 
Pay close attention to tables, lists, and headers."""


def upload_to_gcs(local_file_path, bucket_name):
    """Uploads the file to GCS if it is not already present."""
    bucket = storage_client.bucket(bucket_name)
    blob_name = os.path.basename(local_file_path)
    blob = bucket.blob(blob_name)

    if blob.exists():
        print(f"File already exists in GCS: gs://{bucket_name}/{blob_name}")
        return f"gs://{bucket_name}/{blob_name}"

    print(f"Uploading {local_file_path} to GCS...")
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded successfully: gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


def extract_text_from_pdf(gcs_pdf_uri):
    """Uses Gemini AI to extract text from a PDF stored in GCS."""
    pdf_file = Part.from_uri(file_uri=gcs_pdf_uri, mime_type="application/pdf")

    response = client.models.generate_content(
        model=model,
        contents=[
            Part.from_text(text=SYSTEM_PROMPT),  # System prompt
            Part.from_text(text=USER_PROMPT),  # User prompt
            pdf_file,  # PDF file
        ],
    )
    print(response)
    return response.text


def process_pdf(pdf_path):
    """Uploads a PDF to GCS (if not already uploaded), extracts text, and saves it locally."""
    gcs_uri = upload_to_gcs(pdf_path, BUCKET_NAME)
    extracted_text = extract_text_from_pdf(gcs_uri)

    output_file = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path).replace(".pdf", "_Gemini.txt"))
    if os.path.exists(output_file):
        print(f"Text file '{output_file}' already exists. Skipping...\n\n")
        return
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Text extracted and saved to: {output_file}")


# Process all PDFs in the directory
if __name__ == "__main__":
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the directory.")
    else:
        for pdf_file in pdf_files:
            process_pdf(pdf_file)
