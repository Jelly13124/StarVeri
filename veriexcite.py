# --- NEW: Added fitz for better PDF parsing
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
import requests
import os
import pandas as pd
import re
from unidecode import unidecode
import logging
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai.types import Tool, GoogleSearch, ThinkingConfig
from bs4 import BeautifulSoup
from enum import Enum

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
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text


def extract_bibliography_section(text: str, keywords: List[str] = [
    # English
    "Reference", "References", "Bibliography", "Works Cited",
    # Chinese
    "参考文献", "參考文獻",
    # Japanese
    "参考資料",
    # French
    "Références", "Bibliographie",
    # German
    "Literaturverzeichnis", "Quellenverzeichnis",
    # Spanish
    "Referencias", "Bibliografía",
    # Russian
    "Список литературы",
    # Italian
    "Riferimenti", "Bibliografia",
    # Portuguese
    "Referências", "Bibliografia",
    # Korean
    "참고문헌"
]) -> str:
    """
    Find the last occurrence of any keyword from 'keywords'
    and return the text from that point onward.
    """
    last_index = -1
    for keyword in keywords:
        index = text.lower().rfind(keyword.lower())
        if index > last_index:
            last_index = index
    if last_index == -1:
        raise ValueError("No bibliography section found using keywords: " + ", ".join(keywords))
    return text[last_index:]


# --- Step 2: Split the bibliography text into individual references ---
class ReferenceExtraction(BaseModel):
    title: str
    author: str
    year: int
    bib: str

# This Pydantic model defines the structured output we want from our verification function
class VerificationResult(BaseModel):
    status: ReferenceStatus = Field(description="The verification status: 'validated' or 'not_found'.")
    explanation: str = Field(description="A brief explanation for the status.")
    url: str = Field(description="The canonical URL to the paper if found, otherwise an empty string.")


class ReferenceStatus(Enum):
    VALIDATED = "validated"
    NOT_FOUND = "not_found"


def split_references(bib_text):
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


# --- Step 3: UPGRADED Verification using a single, intelligent function ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def verify_reference_with_search(ref: ReferenceExtraction) -> VerificationResult:
    """
    UPGRADED: Verifies a single reference using an LLM with Google Search tool.
    This replaces all previous individual API calls (Crossref, arXiv, etc.).
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
            model='gemini-1.5-pro', # Using a more powerful model for better reasoning
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
            explanation=f"The verification process encountered an error: {e}",
            url=""
        )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def find_replacement_reference(ref: ReferenceExtraction) -> str:
    """Finds a plausible replacement for an invalid reference using the Gemini API."""
    try:
        prompt = f"""
        The following academic reference was found to be invalid or non-existent:
        Title: "{ref.title}"
        Author: {ref.author}
        Year: {ref.year}

        Please find a single, real, verifiable academic reference (like a journal article, conference paper, or book) that is a plausible replacement. The replacement should be on a very similar topic.

        Provide only the full, correctly formatted bibliographic entry for the single best replacement reference.
        If you cannot find a suitable replacement, return the exact string "No suitable replacement found."
        """

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config={'temperature': 0.2},
        )
        replacement_text = response.text.strip().replace('\n', ' ')
        return replacement_text

    except Exception as e:
        logging.error(f"Failed to find replacement for '{ref.title}': {e}")
        return "Failed to search for a replacement due to an error."


# --- Main Workflow ---

def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Main workflow to check references in a PDF.
    """
    # 1. Extract text and bibliography
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)

    # 2. Split into individual references
    references = split_references(bib_text)

    # 3. Verify each reference
    count_verified, count_warning = 0, 0
    list_warning = []
    list_explanations = []

    for ref in references:
        result = verify_reference_with_search(ref) # Use the new, unified function
        
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

def process_pdf_file(pdf_path: str):
    """Check a single PDF file and print results."""
    count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
    print(f"\n--- Results for {os.path.basename(pdf_path)} ---")
    print(f"✅ Verified References: {count_verified}")
    print(f"⚠️ Warnings (Not Found): {count_warning}")
    
    if count_warning > 0:
        print("\n--- Details for Unverified References ---")
    
    for explanation in list_explanations:
        if "Status: ⚠️" in explanation:
            print(explanation)
            print("-" * 20)

# ... (The process_folder function can remain the same) ...

if __name__ == "__main__":
    # You will need to install PyMuPDF: pip install pymupdf
    
    ''' Set your Google Gemini API key here '''
    GOOGLE_API_KEY = "YOUR_API_KEY"
    if GOOGLE_API_KEY == "YOUR_API_KEY":
        print("Please set your GOOGLE_API_KEY in the script.")
    else:
        set_google_api_key(GOOGLE_API_KEY)

        ''' Example usage: check a single PDF file '''
        pdf_path = "path/to/your/paper.pdf"
        if os.path.exists(pdf_path):
            process_pdf_file(pdf_path)
        else:
            print(f"File not found: {pdf_path}. Please update the path.")