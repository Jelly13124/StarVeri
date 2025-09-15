import fitz  # PyMuPDF
from pydantic import BaseModel, Field
import os
import pandas as pd
import logging
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai.types import Tool, GoogleSearch
from enum import Enum
import re

# --- Configuration ---
GOOGLE_API_KEY = None

def set_google_api_key(api_key: str):
    """Set Google Gemini API key."""
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key


# --- Step 1: UPGRADED PDF and bibliography extraction ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    UPGRADED: Extract text from all pages of the PDF using PyMuPDF (fitz) for better accuracy.
    """
    text = ""
    # Ensure you have installed PyMuPDF: pip install pymupdf
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        raise IOError(f"Error processing PDF file '{pdf_path}': {e}")
    return text


def extract_bibliography_section(text: str, keywords: List[str] = [
    # A comprehensive list of keywords for finding the bibliography
    "Reference", "References", "Bibliography", "Works Cited", "参考文献", "參考文獻",
    "参考資料", "Références", "Bibliographie", "Literaturverzeichnis", "Quellenverzeichnis",
    "Referencias", "Bibliografía", "Список литературы", "Riferimenti", "Bibliografia",
    "Referências", "참고문헌"
]) -> str:
    """
    Find the last occurrence of any keyword from 'keywords' and return the text from that point onward.
    """
    last_index = -1
    for keyword in keywords:
        # Search for the keyword as a whole word at the beginning of a line for more accuracy
        try:
            for match in re.finditer(r'^\s*' + re.escape(keyword) + r'\s*$', text, re.MULTILINE | re.IGNORECASE):
                if match.start() > last_index:
                    last_index = match.start()
        except re.error:
            # Fallback for complex keywords
            index = text.lower().rfind(keyword.lower())
            if index > last_index:
                last_index = index

    if last_index == -1:
        raise ValueError("No bibliography section found using keywords: " + ", ".join(keywords))
    return text[last_index:]


# --- Step 2: Define data structures and split references ---
class ReferenceExtraction(BaseModel):
    title: str
    author: str # First author's last name or organization name
    year: int
    bib: str   # The full, cleaned bibliographic string

class ReferenceStatus(Enum):
    VALIDATED = "validated"
    NOT_FOUND = "not_found"

# Pydantic model for the structured output from our verification function
class VerificationResult(BaseModel):
    status: ReferenceStatus = Field(description="The verification status: 'validated' or 'not_found'.")
    explanation: str = Field(description="A brief explanation for the status.")
    url: str = Field(description="The canonical URL to the paper if found, otherwise an empty string.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def split_references(bib_text: str) -> List[ReferenceExtraction]:
    """Splits the bibliography text into individual references using the Google Gemini API."""
    prompt = """
    Process this bibliography section. Your task is to clean it and split it into a structured list of individual references.
    For each reference, extract the following information:
    1.  **title**: The full title of the work.
    2.  **author**: The last name of the first author. If the author is an organization, use its name.
    3.  **year**: The 4-digit publication year.
    4.  **bib**: The full, cleaned-up, single-line bibliographic string for the reference.

    Correct any obvious OCR or text extraction errors (e.g., weird spacing, broken words).
    Return the data as a JSON list.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt + bib_text,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[ReferenceExtraction],
            'temperature': 0,
        },
    )
    return response.parsed


# --- Step 3: UPGRADED Verification and Replacement Functions ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def verify_reference_with_search(ref: ReferenceExtraction) -> VerificationResult:
    """
    UPGRADED: Verifies a single reference using an LLM with Google Search tool.
    This replaces all previous individual API calls.
    """
    prompt = f"""
    You are an expert academic fact-checker. I will provide a bibliographic reference.
    Your task is to use the Google Search tool to determine if this is a real, published academic work.

    Reference to verify: "{ref.bib}"

    Follow these steps:
    1.  Search for the paper using its title and authors.
    2.  Analyze the search results to find a canonical source like Google Scholar, arXiv, ACM, IEEE, a publisher's site, or a university repository.
    3.  Compare the key details (title, authors, year) from the source with the provided reference. Tolerate minor formatting differences.
    4.  Conclude the verification and provide your response in the requested JSON format.

    - If you find a credible source that matches the reference, set status to 'validated'.
    - If you cannot find any matching academic work after a thorough search, set status to 'not_found'.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    google_search_tool = Tool(google_search=GoogleSearch())
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config={
                'tools': [google_search_tool],
                'response_mime_type': 'application/json',
                'response_schema': VerificationResult,
                'temperature': 0,
            },
        )
        return response.parsed
    except Exception as e:
        logging.error(f"Verification failed for '{ref.title}' with error: {e}")
        return VerificationResult(
            status=ReferenceStatus.NOT_FOUND,
            explanation=f"The verification process encountered an API or parsing error: {e}",
            url=""
        )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def find_replacement_reference(ref: ReferenceExtraction) -> str:
    """Finds a plausible replacement for an invalid reference using the Gemini API."""
    prompt = f"""
    The following academic reference was found to be invalid or non-existent:
    Title: "{ref.title}"
    Author: {ref.author}
    Year: {ref.year}

    Please find a single, real, verifiable academic reference (like a journal article or conference paper) that is a plausible replacement. The replacement should be on a very similar topic.
    Provide only the full, correctly formatted bibliographic entry for the single best replacement reference.
    If you cannot find a suitable replacement, return the exact string "No suitable replacement found."
    """
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config={'temperature': 0.2},
        )
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        logging.error(f"Failed to find replacement for '{ref.title}': {e}")
        return "Failed to search for a replacement due to an error."


# --- Main Workflow ---
def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Main workflow to check references in a PDF using the upgraded, intelligent verification method.
    """
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)
    references = split_references(bib_text)

    count_verified, count_warning = 0, 0
    list_warning, list_explanations = [], []

    for ref in references:
        result = verify_reference_with_search(ref) 
        
        if result.status == ReferenceStatus.VALIDATED:
            count_verified += 1
            list_explanations.append(
                f"Reference: {ref.bib}\n"
                f"Status: ✅ Validated\n"
                f"Explanation: {result.explanation}\n"
                f"Found at: {result.url}\n"
            )
        else:
            count_warning += 1
            list_warning.append(ref.bib)
            replacement_suggestion = find_replacement_reference(ref)
            list_explanations.append(
                f"Reference: {ref.bib}\n"
                f"Status: ⚠️ Not Found\n"
                f"Explanation: {result.explanation}\n"
                f"Suggested Replacement: {replacement_suggestion}\n"
            )
            
    return count_verified, count_warning, list_warning, list_explanations


def process_pdf_file(pdf_path: str) -> None:
    """Check a single PDF file and print a summary."""
    print(f"\n--- Checking file: {os.path.basename(pdf_path)} ---")
    try:
        count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
        print(f"✅ Verified References: {count_verified}")
        print(f"⚠️ Warnings (Not Found): {count_warning}")
        
        if count_warning > 0:
            print("\n--- Detailed Explanations ---")
            for explanation in list_explanations:
                 print(explanation)
                 print("-" * 25)

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(pdf_path)}: {e}")


def process_folder(folder_path: str) -> None:
    """Check all PDF files in a folder and save results to a CSV."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files in the folder.")

    all_results_data = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n--- Checking file: {pdf_file} ---")
        try:
            count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
            all_results_data.append({
                "File": pdf_file,
                "Verified": count_verified,
                "Warnings": count_warning,
                "Warning List": " || ".join(list_warning),
                "Explanations": "\n\n".join(list_explanations)
            })
            print(f"Finished: {count_verified} verified, {count_warning} warnings.")
        except Exception as e:
            print(f"Could not process {pdf_file}. Error: {e}")
            all_results_data.append({
                "File": pdf_file, "Verified": 0, "Warnings": "N/A", 
                "Warning List": "", "Explanations": f"Failed to process: {e}"
            })
    
    if all_results_data:
        pd.DataFrame(all_results_data).to_csv('VeriExCite_results.csv', index=False)
        print("\nResults saved to VeriExCite_results.csv")


if __name__ == "__main__":
    # IMPORTANT: You must install the PyMuPDF library for this to work
    # pip install pymupdf
    
    # Set your Google Gemini API key here
    # Apply for a key at https://ai.google.dev/
    GOOGLE_API_KEY = "YOUR_API_KEY"
    
    if GOOGLE_API_KEY == "YOUR_API_KEY" or not GOOGLE_API_KEY:
        print("ERROR: Please set your GOOGLE_API_KEY in the script.")
    else:
        set_google_api_key(GOOGLE_API_KEY)

        # --- CHOOSE ONE OPTION ---
        
        # Option 1: Check a single PDF file
        # pdf_path = "path/to/your/paper.pdf"
        # if os.path.exists(pdf_path):
        #     process_pdf_file(pdf_path)
        # else:
        #     print(f"File not found: {pdf_path}. Please update the path.")

        # Option 2: Check all PDF files in a folder
        folder_path = "path/to/your/folder"
        if os.path.isdir(folder_path):
            process_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}. Please update the path.")