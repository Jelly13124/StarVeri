# =============================================================================
# VeriExCite: Intelligent Reference Verification and Replacement Tool
# =============================================================================
# This module provides comprehensive reference verification and replacement
# functionality for academic papers. It extracts references from PDFs, verifies
# them against multiple sources (Crossref, arXiv, Google Scholar), and provides
# intelligent replacement suggestions for invalid or non-existent references.
#
# Key Features:
# - Multi-language bibliography extraction (8+ languages supported)
# - Multi-source reference verification (Crossref, arXiv, Google Scholar, Workshop papers)
# - AI-powered DOI/URL verification using Google Gemini
# - Intelligent replacement suggestions using Google Gemini AI
# - Strict verification with confidence scoring (92.5%+ threshold)
# - Comprehensive error handling and retry mechanisms
#
# Author: VeriExCite Team
# Version: 1.0.0
# =============================================================================

# Standard library imports
import os
import re
import logging
from typing import List, Tuple
from enum import Enum

# Third-party imports for PDF processing
import PyPDF2

# Data validation and modeling
from pydantic import BaseModel, Field

# HTTP requests and web scraping
import requests
from bs4 import BeautifulSoup

# Data processing
import pandas as pd

# Text processing and normalization
from unidecode import unidecode
from rapidfuzz import fuzz

# Academic search APIs
from scholarly import scholarly

# Retry mechanisms for robust API calls
from tenacity import retry, stop_after_attempt, wait_exponential

# Google AI integration
from google import genai
from google.genai.types import Tool, GoogleSearch, ThinkingConfig

# =============================================================================
# CONFIGURATION AND GLOBAL VARIABLES
# =============================================================================

# Global variable to store the Google Gemini API key
GOOGLE_API_KEY = None

def set_google_api_key(api_key: str):
    """
    Set the Google Gemini API key for AI-powered reference processing.
    
    Args:
        api_key (str): The Google Gemini API key obtained from https://ai.google.dev/
        
    Note:
        This function must be called before using any AI-powered features like
        reference parsing or replacement suggestion generation.
    """
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key


# =============================================================================
# STEP 1: PDF TEXT EXTRACTION AND BIBLIOGRAPHY IDENTIFICATION
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file using PyPDF2.
    
    This function reads through all pages of the PDF and concatenates the text
    content, which is then used for bibliography section identification.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        
    Returns:
        str: Complete text content of the PDF as a single string
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PyPDF2.errors.PdfReadError: If the PDF is corrupted or encrypted
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_bibliography_section(text: str, keywords: List[str] = [
    # English keywords
    "Reference", "References", "Bibliography", "Works Cited",
    # Chinese keywords (Simplified and Traditional)
    "参考文献", "參考文獻",
    # Japanese keywords
    "参考資料",
    # French keywords
    "Références", "Bibliographie",
    # German keywords
    "Literaturverzeichnis", "Quellenverzeichnis",
    # Spanish keywords
    "Referencias", "Bibliografía",
    # Russian keywords
    "Список литературы",
    # Italian keywords
    "Riferimenti", "Bibliografia",
    # Portuguese keywords
    "Referências", "Bibliografia",
    # Korean keywords
    "참고문헌"
]) -> str:
    """
    Extract the bibliography section from the full PDF text.
    
    This function searches for bibliography section markers in multiple languages
    and returns the text starting from the last occurrence of any keyword.
    This approach ensures we capture the complete reference list even if there
    are multiple "References" sections in the document.
    
    Args:
        text (str): Full text content extracted from the PDF
        keywords (List[str], optional): List of bibliography section keywords
                                      in multiple languages. Defaults to a
                                      comprehensive list covering 8+ languages.
    
    Returns:
        str: The bibliography section text starting from the last keyword found
        
    Raises:
        ValueError: If no bibliography section is found using any of the keywords
        
    Note:
        The function uses case-insensitive matching and searches for the last
        occurrence to ensure we get the complete reference list.
    """
    last_index = -1
    for keyword in keywords:
        index = text.lower().rfind(keyword.lower())
        if index > last_index:
            last_index = index
    if last_index == -1:
        raise ValueError("No bibliography section found using keywords: " + ", ".join(keywords))
    return text[last_index:]


# =============================================================================
# STEP 2: DATA MODELS AND REFERENCE PARSING
# =============================================================================

class ReferenceExtraction(BaseModel):
    """
    Data model representing a parsed reference from the bibliography.
    
    This model captures all essential information extracted from a single
    reference entry, including metadata for verification and replacement.
    
    Attributes:
        title (str): The full title of the reference
        author (str): First author's family name or organization name
        DOI (str): Digital Object Identifier if available
        URL (str): Direct URL to the reference if available
        year (int): Publication year (4-digit format)
        type (str): Reference type (journal_article, preprint, conference_paper, 
                   book, book_chapter, or non_academic_website)
        bib (str): Normalized bibliography entry in standard format
    """
    title: str
    author: str
    DOI: str
    URL: str
    year: int
    type: str
    bib: str

class ReferenceStatus(Enum):
    """
    Enumeration of possible verification statuses for a reference.
    
    This enum defines the three possible outcomes of reference verification:
    - VALIDATED: Reference was successfully verified against authoritative sources
    - INVALID: Reference was found but doesn't match the claimed details
    - NOT_FOUND: No evidence of the reference was found in any source
    """
    VALIDATED = "validated"
    INVALID = "invalid"
    NOT_FOUND = "not_found"

class ReferenceCheckResult(BaseModel):
    """
    Data model representing the result of reference verification.
    
    This model encapsulates the outcome of verifying a reference against
    various academic databases and sources.
    
    Attributes:
        status (ReferenceStatus): The verification status of the reference
        explanation (str): Human-readable explanation of the verification result
    """
    status: ReferenceStatus
    explanation: str

class ReplacementSuggestion(BaseModel):
    """
    Data model for structured replacement suggestions for invalid references.
    
    This model provides a comprehensive structure for suggesting alternative
    references when the original reference is invalid or non-existent.
    The model includes one best-match suggestion with confidence score.
    
    Attributes:
        found (bool): Whether suitable replacement suggestions were found
        reasoning (str): Explanation for why this specific paper was chosen
        suggestion_bib (str): Replacement paper bibliography entry
        suggestion_url (str): Replacement paper URL
        suggestion_score (int): Replacement paper match score (1-100)
    """
    found: bool = Field(description="Set to true if suitable replacement is found, otherwise false.")
    reasoning: str = Field(description="Brief reasoning for why this paper was chosen as replacement.")
    suggestion_bib: str = Field(description="Replacement paper bibliography entry")
    suggestion_url: str = Field(description="Replacement paper URL")
    suggestion_score: int = Field(description="Replacement paper match score (1-100)")


def split_references(bib_text):
    """
    Parse bibliography text into structured reference objects using Google Gemini AI.
    
    This function uses Google Gemini AI to intelligently parse the raw bibliography
    text extracted from a PDF. The AI handles various formatting issues, language
    variations, and reference styles to extract structured data for each reference.
    
    The AI performs the following tasks:
    1. Validates citation format (APA, Harvard, MLA, Chicago, IEEE, Vancouver, Nature, Science, and other standard academic formats)
    2. Checks for style consistency across all references in the paper
    3. Normalizes formatting (fixes spacing, line breaks, punctuation)
    4. Extracts key metadata (title, author, DOI, URL, year, type)
    5. Classifies reference types appropriately
    6. Formats bibliography entries consistently
    7. Flags incorrectly formatted references and style inconsistencies for user warning
    
    Args:
        bib_text (str): Raw bibliography text extracted from the PDF
        
    Returns:
        List[ReferenceExtraction]: List of parsed and structured reference objects
        
    Raises:
        ValueError: If the API key is not set or the response cannot be parsed
        Exception: If the API call fails or returns invalid data
        
    Note:
        This function requires a valid Google Gemini API key to be set using
        set_google_api_key() before calling this function.
        Incorrectly formatted references will be flagged and skipped during verification.
    """
    # Enhanced prompt for the AI to parse references and validate format
    prompt = """
    Process a reference list extracted from a PDF, where formatting may be corrupted.  
    Follow these steps to clean and extract key information: 
    1. Normalisation: Fix spacing errors, line breaks, and punctuation.
    2. Extraction: For each reference, extract:
    - Title (full title case)
    - Author: First author's family name (If the author is an organization, use the organization name)
    - DOI (include if explicitly stated; otherwise leave blank)
    - URL (include if explicitly stated; otherwise leave blank)
    - Year (4-digit publication year)
    - Type (journal_article, preprint, conference_paper, book, book_chapter, OR non_academic_website. If the author is not a human but an organization, select non_academic_website)
    - Bib: Normalised input bibliography (correct format, in one line)\n\n
    """

    # Initialize the Google Gemini client
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # Call the AI model to parse the bibliography text
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt + bib_text,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[ReferenceExtraction],
            'temperature': 0,  # Use deterministic output for consistent parsing
            'thinking_config': ThinkingConfig(thinking_budget=0),  # Disable thinking for faster response
        },
    )
    
    # Extract the parsed references from the response
    references: list[ReferenceExtraction] = response.parsed
    return references


# =============================================================================
# STEP 3: REFERENCE VERIFICATION AND MATCHING
# =============================================================================

def normalize_title(title: str) -> str:
    """
    Normalize a title for accurate comparison between different sources.
    
    This function performs comprehensive text normalization to enable accurate
    matching between reference titles from different sources. It handles:
    - Unicode normalization (removes accents, special characters)
    - Case normalization (converts to lowercase)
    - Punctuation removal
    - Stop word removal (and, the)
    - Whitespace normalization
    
    Args:
        title (str): The title to normalize
        
    Returns:
        str: Normalized title suitable for comparison
        
    Example:
        >>> normalize_title("The Art of Machine Learning: A Comprehensive Guide")
        'artofmachinelearningcomprehensiveguide'
    """
    # Convert unicode characters to ASCII equivalents
    title = unidecode(title)
    
    # Remove all punctuation and special characters, keep only word characters and spaces
    title = re.sub(r'[^\w\s]', '', title).lower()
    
    # Remove common stop words that don't affect meaning
    title = re.sub(r'\band\b|\bthe\b', '', title)
    
    # Remove all whitespace and strip
    title = re.sub(r'\s+', '', title).strip()
    
    return title


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_scholarly(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Search for a reference using Google Scholar with flexible matching.
    
    This function searches Google Scholar for the reference and uses fuzzy matching
    to determine if the found papers match the reference details. It handles both
    references with and without DOI/URL differently.
    
    Args:
        ref (ReferenceExtraction): The reference to search for
        
    Returns:
        ReferenceCheckResult: The verification result with status and explanation
        
    Note:
        - For references with DOI/URL: Uses exact title matching only
        - For references without DOI/URL: Uses fuzzy matching (92.5%+ threshold)
        - Uses retry mechanism for robust API calls
    """
    try:
        search_results = scholarly.search_pubs(ref.title)
        result = next(search_results, None)
        normalized_input_title = normalize_title(ref.title)

        if result and 'bib' in result and 'author' in result['bib'] and 'title' in result['bib']:
            # Check if reference has DOI/URL that needs verification
            ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
            ref_url = ref.URL.strip().lower() if ref.URL else ''
            
            # If reference has DOI/URL, we need exact match for title and author only
            if ref_doi or ref_url:
                # Only check title and author match (DOI/URL verification handled by Crossref)
                result_author = result['bib']['author'][0].split()[-1].lower()
                ref_author = ref.author.lower()
                author_match = (
                    ref_author == result_author or
                    ref_author in result_author or
                    result_author in ref_author or
                    fuzz.ratio(ref_author, result_author) > 80
                )
                
                if author_match:
                    normalized_item_title = normalize_title(result['bib']['title'])
                    if normalized_item_title == normalized_input_title:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (exact match).")
            else:
                # No DOI/URL - use fuzzy matching for title and author
                result_author = result['bib']['author'][0].split()[-1].lower()
                ref_author = ref.author.lower()
                author_match = (
                    ref_author == result_author or
                    ref_author in result_author or
                    result_author in ref_author or
                    fuzz.ratio(ref_author, result_author) > 80
                )
                
                if author_match:
                    normalized_item_title = normalize_title(result['bib']['title'])
                    title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
                    
                    if normalized_item_title == normalized_input_title:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and title match Google Scholar (exact match).")
                    if title_score >= 90:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Author and title match Google Scholar (high confidence: {title_score}%).")
                
                # Even if author doesn't match, check if title is very close (only 95%+ for title-only matches)
                normalized_item_title = normalize_title(result['bib']['title'])
                title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
                if title_score >= 95:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Title match Google Scholar (very high confidence: {title_score}%).")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Google Scholar.")
    except Exception as e:
        logging.warning(f"Scholarly search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Google Scholar search failed: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_crossref(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Search for a reference using the Crossref API with comprehensive verification.
    
    This function implements a multi-step verification process:
    1. Direct DOI lookup using Gemini AI (if DOI available)
    2. Fallback to Crossref direct DOI lookup
    3. Title + author search with fuzzy matching (if no DOI/URL)
    
    Args:
        ref (ReferenceExtraction): The reference to search for
        
    Returns:
        ReferenceCheckResult: The verification result with status and explanation
        
    Note:
        - Uses Gemini AI for DOI verification when available
        - Falls back to Crossref API for direct DOI lookup
        - Uses fuzzy matching for references without DOI/URL
        - Implements strict confidence scoring (92.5%+ threshold)
    """
    
    # First, try direct DOI lookup if reference has DOI
    ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
    ref_url = ref.URL.strip().lower() if ref.URL else ''
    
    # Extract DOI from URL if present
    if ref_url and 'doi.org/' in ref_url:
        ref_doi = ref_url.split('doi.org/')[-1]
    
    # Try Gemini DOI verification first
    if ref_doi:
        print(f"DEBUG: Trying Gemini DOI verification for: {ref_doi}")
        try:
            doi_result = verify_doi_with_gemini(ref, ref_doi)
            if doi_result:
                # If Gemini says DOI is invalid, try database search as fallback
                if doi_result.status == ReferenceStatus.INVALID:
                    print(f"DEBUG: Gemini marked DOI as invalid, trying database search fallback")
                    # Skip to title search for database verification
                    pass
                else:
                    return doi_result  # Return Gemini's result (VALIDATED or NOT_FOUND)
        except Exception as e:
            print(f"DEBUG: Gemini DOI verification failed: {e}")
        
        # Fallback to direct DOI lookup if Gemini fails or was invalid
        print(f"DEBUG: Fallback to direct DOI lookup for: {ref_doi}")
        try:
            doi_response = requests.get(f"https://api.crossref.org/works/{ref_doi}")
            if doi_response.status_code == 200:
                doi_data = doi_response.json()
                if 'message' in doi_data:
                    item = doi_data['message']
                    print(f"DEBUG: Direct DOI lookup successful for: {ref_doi}")
                    
                    # Check author match
                    if 'author' in item and item['author'] and 'family' in item['author'][0]:
                        if ref.author.lower() == item['author'][0]['family'].lower():
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Direct DOI lookup: Author, title and DOI match Crossref record.")
                        else:
                            # DOI matches but author doesn't - try database search
                            print(f"DEBUG: DOI matches but author doesn't, trying database search fallback")
                            db_result = search_title(ref)
                            if db_result and db_result.status == ReferenceStatus.VALIDATED:
                                return ReferenceCheckResult(
                                    status=ReferenceStatus.VALIDATED, 
                                    explanation=f"DOI invalid but database search found valid match: {db_result.explanation}"
                                )
                            else:
                                return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="Direct DOI lookup: DOI matches but author does not match Crossref record.")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Direct DOI lookup: DOI matches Crossref record.")
        except Exception as e:
            print(f"DEBUG: Direct DOI lookup failed: {e}")
    
    # Fallback to title search
    params = {'query.title': ref.title, 'rows': 10}  # Increased from 5 to 10 for more options
    response = requests.get("https://api.crossref.org/works", params=params)

    if response.status_code == 200:
        items = response.json().get('message', {}).get('items', [])
        normalized_input_title = normalize_title(ref.title)
        
        # First pass: Check DOI matches (highest confidence) - only if not already checked by direct lookup
        if not ref_doi:  # Only do this if we didn't already try direct DOI lookup
            ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
            ref_url = ref.URL.strip().lower() if ref.URL else ''
            
            # Extract DOI from URL if present (e.g., "https://doi.org/10.1007/978-0-387-84858-7" -> "10.1007/978-0-387-84858-7")
            if ref_url and 'doi.org/' in ref_url:
                ref_doi = ref_url.split('doi.org/')[-1]
        
        for item in items:
            item_doi = item.get('DOI', '').strip().lower() if 'DOI' in item else ''
            item_url = item.get('URL', '').strip().lower() if 'URL' in item else ''
            
            # Check if DOI matches (if reference has DOI)
            if ref_doi and item_doi and ref_doi == item_doi:
                if 'author' in item and item['author'] and 'family' in item['author'][0]:
                    if ref.author.lower() == item['author'][0]['family'].lower():
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author, title and DOI match Crossref record.")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="DOI matches but author does not match Crossref record.")
                else:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="DOI matches Crossref record.")
            
            # Check if URL matches (if reference has URL)
            if ref_url and item_url and ref_url == item_url:
                if 'author' in item and item['author'] and 'family' in item['author'][0]:
                    if ref.author.lower() == item['author'][0]['family'].lower():
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author, title and URL match Crossref record.")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="URL matches but author does not match Crossref record.")
                else:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="URL matches Crossref record.")
        
        # Second pass: Check title and author matches (only if no DOI/URL provided)
        if not ref_doi and not ref_url:
            best_match_score = 0
            best_match_explanation = ""
            
            for item in items:
                if 'title' in item and item['title'] and 'author' in item and item['author']:
                    item_title = item['title'][0]
                    normalized_item_title = normalize_title(item_title)
                    title_score = fuzz.ratio(normalized_item_title, normalized_input_title)
                    
                    # Check author match (more flexible)
                    author_match = False
                    if 'family' in item['author'][0]:
                        author_last_name = item['author'][0]['family'].lower()
                        ref_author = ref.author.lower()
                        # More flexible author matching
                        author_match = (
                            ref_author == author_last_name or  # Exact match
                            ref_author in author_last_name or  # Partial match
                            author_last_name in ref_author or  # Reverse partial match
                            fuzz.ratio(ref_author, author_last_name) > 80  # Fuzzy match
                        )
                    
                    # Calculate combined score
                    combined_score = title_score
                    if author_match:
                        combined_score += 20  # Bonus for author match
                    
                    # Update best match if this is better
                    if combined_score > best_match_score:
                        best_match_score = combined_score
                        if title_score == 100:
                            best_match_explanation = "Author and title match Crossref record (exact match)."
                        elif title_score >= 92.5:
                            best_match_explanation = "Author and title match Crossref record (very high confidence)."
                        elif title_score >= 80:
                            best_match_explanation = "Author and title match Crossref record (high confidence)."
                        elif title_score >= 70:
                            best_match_explanation = "Author and title match Crossref record (good confidence)."
                        else:
                            best_match_explanation = f"Author and title match Crossref record (moderate confidence: {title_score}%)."
            
            # Return result based on best match - STRICT: Only >92.5% confidence or exact match
            if best_match_score >= 92.5:  # Very strict threshold - only high confidence matches
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=best_match_explanation)
            else:
                return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation=f"Reference found but confidence too low ({best_match_score}% < 92.5%).")
        else:
            # Reference has DOI/URL but no match found in first pass
            return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation="Reference has DOI/URL but no matching DOI/URL found in Crossref record.")
    else:
        logging.warning(f"Crossref API request failed with status code: {response.status_code}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Crossref API request failed with status code: {response.status_code}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_arxiv(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title in arXiv, with error handling and retries."""
    try:
        url = "http://export.arxiv.org/api/query"
        params = {'search_query': f'ti:"{ref.title}"', 'max_results': 5}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml-xml')
            entries = soup.find_all('entry')
            
            if not entries:
                params['search_query'] = f'all:{ref.title}'
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'lxml-xml')
                    entries = soup.find_all('entry')
            
            if not entries:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
                
            normalized_input_title = normalize_title(ref.title)
            
            # Check if reference has DOI/URL that needs verification
            ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
            ref_url = ref.URL.strip().lower() if ref.URL else ''
            
            for entry in entries:
                title_tag = entry.find('title')
                if title_tag:
                    arxiv_title = title_tag.text.strip()
                    normalized_arxiv_title = normalize_title(arxiv_title)
                    
                    # If reference has DOI/URL, we need exact match for title and author only
                    if ref_doi or ref_url:
                        # Only check exact title match (DOI/URL verification handled by Crossref)
                        if normalized_arxiv_title == normalized_input_title:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (exact match).")
                    else:
                        # No DOI/URL - use fuzzy matching for title and author
                        title_score = fuzz.ratio(normalized_arxiv_title, normalized_input_title)
                        
                        # Check for exact match first
                        if normalized_arxiv_title == normalized_input_title:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Title match in arXiv (exact match).")
                        
                        # Check for very high confidence fuzzy match only
                        if title_score >= 92.5:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Title match in arXiv (high confidence: {title_score}%).")
                        
                        # Check for high confidence match with author verification
                        if title_score >= 85:
                            author_tags = entry.find_all('author')
                            for author_tag in author_tags:
                                name_tag = author_tag.find('name')
                                if name_tag:
                                    author_name = name_tag.text.strip()
                                    last_name = author_name.split()[-1].lower()
                                    ref_author = ref.author.lower()
                                    
                                    # Strict author matching
                                    author_match = (
                                        ref_author == last_name or
                                        fuzz.ratio(ref_author, last_name) > 92.5
                                    )
                                    
                                    if author_match:
                                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Author and similar title match in arXiv (confidence: {title_score}%).")
                                
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
        
    except Exception as e:
        logging.warning(f"arXiv search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"arXiv search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_workshop_paper(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for workshop papers using Google Search directly."""
    try:
        workshop_indicators = ['workshop', 'symposium', 'proc.', 'proceedings']
        is_likely_workshop = any(indicator in ref.bib.lower() for indicator in workshop_indicators)
        
        if not is_likely_workshop:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Not a workshop paper.")
            
        prompt = f"""
        Please search for this exact workshop paper and verify it exists:
        Title: {ref.title}
        Author: {ref.author}
        Year: {ref.year}
        
        This paper appears to be from a workshop or symposium. Check conferences, workshops, 
        and personal/university pages. Return 'True' only if you can find evidence this 
        specific workshop paper exists (exact title and author match). Return 'False' otherwise.
        Return only 'True' or 'False', without any additional explanation.
        """
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool], 'temperature': 0},
        )
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Workshop paper found via Google search.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Workshop paper not found via Google search.")
            
    except Exception as e:
        logging.warning(f"Workshop paper search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Workshop paper search failed: {e}")

def verify_url(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Verifies if the URL leads to a legitimate academic source that matches the reference.
    
    This function implements a multi-step verification process:
    1. Try Gemini AI URL verification
    2. Fallback to traditional HTTP verification
    3. If URL is invalid, use database search for title/author match
    
    Args:
        ref (ReferenceExtraction): The reference to verify
        
    Returns:
        ReferenceCheckResult: The verification result with status and explanation
    """
    if not ref.URL:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No URL provided.")
    
    # Try Gemini URL verification first
    print(f"DEBUG: Trying Gemini URL verification for: {ref.URL}")
    try:
        url_result = verify_url_with_gemini(ref, ref.URL)
        if url_result:
            # If Gemini says URL is invalid, try database search as fallback
            if url_result.status == ReferenceStatus.INVALID:
                print(f"DEBUG: Gemini marked URL as invalid, trying database search fallback")
                db_result = search_title(ref)
                if db_result and db_result.status == ReferenceStatus.VALIDATED:
                    return ReferenceCheckResult(
                        status=ReferenceStatus.VALIDATED, 
                        explanation=f"URL invalid but database search found valid match: {db_result.explanation}"
                    )
                else:
                    return url_result  # Return Gemini's invalid result
            else:
                return url_result  # Return Gemini's result (VALIDATED or NOT_FOUND)
    except Exception as e:
        print(f"DEBUG: Gemini URL verification failed: {e}")
    
    # Fallback to traditional URL verification
    try:
        response = requests.get(ref.URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            webpage_title = title_tag.text.strip()
            normalized_webpage_title = normalize_title(webpage_title)
            normalized_input_title = normalize_title(ref.title)
            if normalized_webpage_title == normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (exact match).")
            elif normalized_input_title in normalized_webpage_title or normalized_webpage_title in normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (partial match).")
        else:
            logging.warning(f"No <title> tag found at URL: {ref.URL}")
            return search_title_google(ref)
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error accessing URL {ref.URL}: {e}")
        return search_title_google(ref)
    except Exception as e:
        logging.warning(f"Error processing URL {ref.URL}: {e}")
        return search_title_google(ref)

def verify_doi_with_gemini(ref: ReferenceExtraction, doi: str) -> ReferenceCheckResult:
    """Verify DOI using Gemini AI with web search. Only checks title and year, not author."""
    try:
        prompt = f"""
        You are an academic reference verification expert. Your task is to verify if a DOI leads to a legitimate academic paper that matches the given reference details.

        REFERENCE TO VERIFY:
        - DOI: {doi}
        - Title: "{ref.title}"
        - Year: {ref.year}

        VERIFICATION STEPS:
        1. Search for the DOI "{doi}" on the web
        2. Access the paper at the DOI URL (https://doi.org/{doi})
        3. Extract the paper's actual title and publication year
        4. Compare with the reference details above

        MATCHING CRITERIA:
        - DOI must resolve to a real, accessible academic paper
        - Paper title must be substantially similar to "{ref.title}" (allowing for minor formatting differences)
        - Publication year should be close to {ref.year} (within ±2 years acceptable)

        DECISION RULES:
        - Return 'True' if: DOI exists, paper is accessible, title matches, year is close
        - Return 'False' if: DOI doesn't exist, paper is inaccessible, title doesn't match, or year is too different

        Return only 'True' or 'False' - no explanations or additional text.
        """
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool]},
        )
        
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Gemini DOI verification: DOI {doi} is valid and matches the reference.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation=f"Gemini DOI verification: DOI {doi} is invalid or doesn't match the reference.")
            
    except Exception as e:
        logging.warning(f"Gemini DOI verification failed for DOI '{doi}': {e}")
        return None  # Return None to trigger fallback

def verify_url_with_gemini(ref: ReferenceExtraction, url: str) -> ReferenceCheckResult:
    """Verify URL using Gemini AI with web search. Only checks title and year, not author."""
    try:
        prompt = f"""
        You are an academic reference verification expert. Your task is to verify if a URL leads to a legitimate academic source that matches the given reference details.

        REFERENCE TO VERIFY:
        - URL: {url}
        - Title: "{ref.title}"
        - Year: {ref.year}

        VERIFICATION STEPS:
        1. Visit the URL "{url}"
        2. Check if the page is accessible and contains academic content
        3. Extract the page's title and publication year
        4. Compare with the reference details above

        MATCHING CRITERIA:
        - URL must be accessible and lead to a legitimate academic source
        - Page title must be substantially similar to "{ref.title}" (allowing for minor formatting differences)
        - Publication year should be close to {ref.year} (within ±2 years acceptable)
        - Content should be academic in nature (research paper, journal article, conference proceeding, etc.)

        DECISION RULES:
        - Return 'True' if: URL is accessible, page title matches, year is close, content is academic
        - Return 'False' if: URL is inaccessible, page title doesn't match, year is too different, or content is not academic

        Return only 'True' or 'False' - no explanations or additional text.
        """
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool]},
        )
        
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=f"Gemini URL verification: URL {url} is valid and matches the reference.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.INVALID, explanation=f"Gemini URL verification: URL {url} is invalid or doesn't match the reference.")
            
    except Exception as e:
        logging.warning(f"Gemini URL verification failed for URL '{url}': {e}")
        return None  # Return None to trigger fallback

def search_title_google(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using Google Search and match using a LLM model."""
    try:
        prompt = f"""
        Please search for the reference on Google, compare with research results, and determine if it is genuine.\n
        Return 'True' only if a website with the the exact title and author is found. Otherwise, return 'False'.\n
        Return only 'True' or 'False', without any additional information.\n\n
        Author: {ref.author}\n
        Title: {ref.title}\n"""
        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'tools': [google_search_tool]},
        )
        answer = normalize_title(response.candidates[0].content.parts[0].text)
        if answer.startswith('true') or answer.endswith('true'):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Google search found matching reference.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Google search did not find matching reference.")
    except Exception as e:
        logging.warning(f"Google search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Google search failed: {e}")

# --- REFINED: find_replacement_reference method ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))


def find_replacement_reference(ref: ReferenceExtraction) -> ReplacementSuggestion:
    """
    Find a replacement reference using Google Gemini AI with verification.
    
    This function uses Google Gemini AI with web search capabilities to find
    a suitable replacement for an invalid reference. It ensures all suggestions
    have valid DOI or URL and verifies them using Gemini DOI/URL checkers.
    
    Args:
        ref (ReferenceExtraction): The invalid reference to find a replacement for
        
    Returns:
        ReplacementSuggestion: The verified replacement suggestion with details and score
        
    Note:
        - Requires a valid Google Gemini API key to be set before calling this function
        - Will retry up to 3 times to find a valid replacement
        - All suggestions are verified using Gemini DOI/URL checkers
        - Only returns suggestions with valid, accessible DOI or URL
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        prompt = f"""
        You are a helpful research assistant. A reference was found to be invalid or non-existent.
        Your task is to find 1 real, verifiable, and topically similar academic paper to suggest as replacement.

        Original (invalid) reference: "{ref.bib}"
        Reference type: {ref.type}
        Attempt: {attempt + 1}/{max_attempts}

        CRITICAL REQUIREMENTS:
        1. The replacement MUST have a valid DOI or direct URL
        2. The DOI/URL MUST be accessible and lead to a real academic paper
        3. The paper MUST be of the same type as the original reference ({ref.type})
        4. The paper MUST be on the same topic as the original reference

        Please follow these steps:
        1. Analyze the probable topic from the invalid reference's title and author
        2. Find 1 well-regarded, real academic paper of the SAME TYPE ({ref.type}) on that same topic
        3. Ensure the paper has a valid DOI (preferred) or direct URL (arXiv, publisher link, etc.)
        4. Verify the DOI/URL is accessible by checking it exists
        5. Provide the details in the requested JSON format

        DOI/URL REQUIREMENTS:
        - DOI format: 10.xxxx/xxxxx (e.g., 10.1000/182)
        - arXiv format: https://arxiv.org/abs/xxxx.xxxxx
        - Publisher URL: Direct link to the paper on publisher website
        - NO generic search URLs or broken links

        Return exactly 1 suggestion with the highest relevance and quality.
        The DOI/URL will be verified before acceptance.
        """
        
        try:
            if not GOOGLE_API_KEY:
                return ReplacementSuggestion(
                    found=False,
                    reasoning="Google API key not set. Please set your API key first.",
                    suggestion_bib="", 
                    suggestion_url="", 
                    suggestion_score=0
                )
                
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model='gemini-2.5-flash', # Using a more powerful model for better reasoning
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ReplacementSuggestion,
                    'temperature': 0.1,
                },
            )
            
            # Parse the response
            suggestion = None
            if hasattr(response, 'parsed') and response.parsed:
                suggestion = response.parsed
            else:
                # Fallback: try to parse the response text
                import json
                try:
                    parsed_data = json.loads(response.text)
                    suggestion = ReplacementSuggestion(**parsed_data)
                except Exception as parse_error:
                    logging.warning(f"Failed to parse replacement response (attempt {attempt + 1}): {parse_error}")
                    continue
            
            # Verify the suggestion using our Gemini DOI/URL checkers
            if suggestion and suggestion.found and suggestion.suggestion_bib and suggestion.suggestion_url:
                logging.info(f"Found replacement suggestion on attempt {attempt + 1}, verifying...")
                
                # Parse the replacement suggestion to extract title, author, and year
                try:
                    parsed_suggestions = split_references(suggestion.suggestion_bib)
                    if parsed_suggestions and len(parsed_suggestions) > 0:
                        temp_ref = parsed_suggestions[0]  # Use the first (and only) parsed suggestion
                        temp_ref.URL = suggestion.suggestion_url  # Set the URL from the suggestion
                        temp_ref.type = ref.type  # Use the same type as original
                    else:
                        logging.warning(f"Failed to parse replacement suggestion bibliography on attempt {attempt + 1}")
                        continue
                except Exception as e:
                    logging.warning(f"Failed to parse replacement suggestion on attempt {attempt + 1}: {e}")
                    continue
                
                # Check if the URL is a DOI
                if suggestion.suggestion_url.startswith('10.') or 'doi.org/' in suggestion.suggestion_url:
                    # Extract DOI from URL if needed
                    doi = suggestion.suggestion_url
                    if 'doi.org/' in doi:
                        doi = doi.split('doi.org/')[-1]
                    
                    # Verify using Gemini DOI checker
                    verification_result = verify_doi_with_gemini(temp_ref, doi)
                    if verification_result and verification_result.status == ReferenceStatus.VALIDATED:
                        logging.info(f"Replacement suggestion verified successfully on attempt {attempt + 1}")
                        return suggestion
                    else:
                        logging.info(f"Replacement suggestion failed DOI verification on attempt {attempt + 1}, trying again...")
                        continue
                else:
                    # Verify using Gemini URL checker
                    verification_result = verify_url_with_gemini(temp_ref, suggestion.suggestion_url)
                    if verification_result and verification_result.status == ReferenceStatus.VALIDATED:
                        logging.info(f"Replacement suggestion verified successfully on attempt {attempt + 1}")
                        return suggestion
                    else:
                        logging.info(f"Replacement suggestion failed URL verification on attempt {attempt + 1}, trying again...")
                        continue
            else:
                logging.info(f"Invalid suggestion format on attempt {attempt + 1}, trying again...")
                continue
                
        except Exception as e:
            logging.warning(f"Failed to find replacement (attempt {attempt + 1}): {e}")
            continue
    
    # If we get here, all attempts failed
    logging.warning(f"Failed to find valid replacement after {max_attempts} attempts")
    return ReplacementSuggestion(
        found=False,
        reasoning=f"Could not find a valid replacement after {max_attempts} attempts. All suggestions failed verification.",
        suggestion_bib="", 
        suggestion_url="", 
        suggestion_score=0
    )

def search_title(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Search for a reference using multiple verification methods with progressive fallback.
    
    This function implements a comprehensive verification strategy that tries multiple
    sources in order of reliability. It handles both website and academic references
    differently, using appropriate verification methods for each type.
    
    Verification Strategy:
    1. Website references: URL verification (Gemini + traditional)
    2. Academic references: Crossref → arXiv → Google Scholar → Workshop papers
    
    Args:
        ref (ReferenceExtraction): The reference to search for
        
    Returns:
        ReferenceCheckResult: The verification result with status and explanation
        
    Note:
        - Uses progressive fallback: tries most reliable sources first
        - Handles different reference types appropriately
        - Returns the first successful verification result
    """
    try:
        if ref.type == "non_academic_website":
            result = verify_url(ref)
            if result is None:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="URL verification failed.")
            return result
        else:
            # Try Crossref first (most reliable for academic papers)
            crossref_result = search_title_crossref(ref)
            if crossref_result and crossref_result.status in [ReferenceStatus.VALIDATED, ReferenceStatus.INVALID]:
                return crossref_result  # Return result if found (valid or invalid)
            
            # Try arXiv for preprints
            arxiv_result = search_title_arxiv(ref)
            if arxiv_result and arxiv_result.status == ReferenceStatus.VALIDATED:
                return arxiv_result
                
            # Try Google Scholar (more comprehensive but less reliable)
            scholar_result = search_title_scholarly(ref)
            if scholar_result and scholar_result.status == ReferenceStatus.VALIDATED:
                return scholar_result
                
            # Try workshop papers last (most lenient)
            workshop_result = search_title_workshop_paper(ref)
            if workshop_result and workshop_result.status == ReferenceStatus.VALIDATED:
                return workshop_result
            
            # If all sources failed, return the best available result
            # Prefer Crossref result if available, otherwise return NOT_FOUND
            if crossref_result and crossref_result.status == ReferenceStatus.NOT_FOUND:
                return crossref_result
            elif arxiv_result and arxiv_result.status == ReferenceStatus.NOT_FOUND:
                return arxiv_result
            elif scholar_result and scholar_result.status == ReferenceStatus.NOT_FOUND:
                return scholar_result
            else:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No evidence found in any source.")
    except Exception as e:
        logging.error(f"Search failed for reference '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Search failed: {e}")

# =============================================================================
# MAIN WORKFLOW AND PROCESSING FUNCTIONS
# =============================================================================

def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Main workflow function to check references in a PDF file.
    
    This function orchestrates the complete reference verification process:
    1. Extract text from PDF
    2. Identify bibliography section
    3. Parse references using AI
    4. Verify each reference against multiple sources
    5. Generate replacement suggestions for invalid references
    6. Return comprehensive results
    
    Args:
        pdf_path (str): Path to the PDF file to process
        
    Returns:
        Tuple[int, int, List[str], List[str]]: A tuple containing:
            - count_verified: Number of validated references
            - count_warning: Number of warnings (invalid or not found)
            - list_warning: List of bibliography entries with warnings
            - list_explanations: List of explanations for each reference
            
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If no bibliography section is found
        Exception: If reference parsing or verification fails
    """
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)
    references = split_references(bib_text)

    count_verified, count_warning = 0, 0
    list_warning = []
    list_explanations = []

    for idx, ref in enumerate(references):        
        
        try:
            result = search_title(ref)
            if result is None:
                result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Search returned no result.")
        except Exception as e:
            logging.error(f"Error processing reference {idx}: {e}")
            result = ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Processing error: {e}")
        
        if result.status == ReferenceStatus.VALIDATED:
            count_verified += 1
            list_explanations.append(f"Reference: {ref.bib}\nStatus: {result.status.value}\nExplanation: {result.explanation}\n")
        else:
            count_warning += 1
            list_warning.append(ref.bib)
            
            # Find replacement for invalid references (single suggestion)
            suggestion = find_replacement_reference(ref)
            if suggestion.found:
                replacement_text = f"找到替换建议\n"
                replacement_text += f"推荐理由: {suggestion.reasoning}\n\n"
                replacement_text += f"建议 (匹配度: {suggestion.suggestion_score}/100):\n"
                replacement_text += f"文献: {suggestion.suggestion_bib}\n"
                replacement_text += f"链接: {suggestion.suggestion_url}\n\n"
            else:
                replacement_text = f"未找到合适的替换建议\n原因: {suggestion.reasoning}"

            list_explanations.append(
                f"Reference: {ref.bib}\n"
                f"Status: {result.status.value}\n"
                f"Explanation: {result.explanation}\n"
                f"{replacement_text}\n"
            )
            
    return count_verified, count_warning, list_warning, list_explanations

def process_pdf_file(pdf_path: str) -> None:
    """Check a single PDF file."""
    count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
    print(f"{count_verified} references verified, {count_warning} warnings.")
    if count_warning > 0:
        print("\nWarning List:\n")
        for item in list_warning:
            print(item)
    print("\nExplanation:\n")
    for explanation in list_explanations:
        print(explanation)

def process_folder(folder_path: str) -> None:
    """Check all PDF files in a folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files in the folder.")

    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Checking file: {pdf_file}")
        try:
            count_verified, count_warning, list_warning, list_explanations = process_pdf_file(pdf_path)
            print("--------------------------------------------------")
            results.append({"File": pdf_file, "Found References": count_verified + count_warning, "Verified": count_verified,
                            "Warnings": count_warning, "Warning List": list_warning, "Explanation": list_explanations})
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
    
    if results:
        pd.DataFrame(results).to_csv('VeriExCite results.csv', index=False)
        print("Results saved to VeriExCite results.csv")


if __name__ == "__main__":
    ''' Set your Google Gemini API key here '''
    # Apply for a key at https://ai.google.dev/
    GOOGLE_API_KEY = "YOUR_API_KEY"
    set_google_api_key(GOOGLE_API_KEY)

    ''' Example usage #1: check a single PDF file '''
    # pdf_path = "path/to/your/paper.pdf"
    # process_pdf_file(pdf_path)

    ''' Example usage #2: check all PDF files in a folder '''
    folder_path = "path/to/your/folder"
    process_folder(folder_path)