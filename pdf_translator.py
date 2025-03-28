#!/usr/bin/env python3
"""
PDF Language Detection and Translation Script using OpenAI API

This standalone script scans a directory for PDF files, detects if they're in German,
and translates them to English using OpenAI's API.

Usage:
    python pdf_translator_openai.py --input_dir path/to/pdfs --output_dir path/to/output [options]

Required packages:
    - langid: pip install langid
    - openai: pip install openai
    - PyPDF2: pip install PyPDF2
    - tqdm: pip install tqdm
"""

import os
import argparse
import re
import time
import shutil
from pathlib import Path
import langid
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter, errors
from tqdm import tqdm
import openai
from datetime import datetime
import logging
import traceback

# Set up argument parser
parser = argparse.ArgumentParser(description='Detect German PDF files and translate them to English using OpenAI')
parser.add_argument('--input_dir', required=True, help='Directory containing PDF files')
parser.add_argument('--output_dir', required=True, help='Directory for translated PDF files')
parser.add_argument('--openai_api_key', help='OpenAI API Key (alternatively, set the OPENAI_API_KEY environment variable)')
parser.add_argument('--openai_model', default='gpt-4o-mini', help='OpenAI model to use for translation')
parser.add_argument('--min_confidence', type=float, default=0.7, help='Minimum confidence for language detection')
parser.add_argument('--sample_size', type=int, default=20, help='Number of text samples to take from each document')
parser.add_argument('--german_threshold', type=float, default=0.4, help='Threshold to consider a document as German')
parser.add_argument('--no_translate', action='store_true', help='Detect language only, don\'t translate')
parser.add_argument('--force_translate', action='store_true', help='Translate all detected German files, even if previously translated')
parser.add_argument('--log_file', help='Path to log file')

def setup_logging(log_file=None):
    """Set up logging to file and console"""
    # Create logger
    logger = logging.getLogger('pdf_translator')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create file handler if log file specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def setup_openai_api(api_key=None):
    """Set up the OpenAI API"""
    # First check environment variable
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    # If still no API key, ask for it
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    
    # Set the API key
    openai.api_key = api_key
    
    # Test the API key
    try:
        # Make a minimal API call to test authentication
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Hello."}],
            max_tokens=5
        )
        logger.info("OpenAI API connection successful")
        return True
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {str(e)}")
        return False

def detect_language(text, min_confidence=0.7):
    """
    Detect the language of a text.
    Returns tuple: (language code, confidence)
    """
    try:
        # For better accuracy, use a sample of text if it's very long
        sample = text[:3000] if len(text) > 3000 else text
        lang, confidence = langid.classify(sample)
        return lang, confidence
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        # Default to English if detection fails
        return 'en', 0.0

def translate_text_with_openai(text, model="gpt-4o-mini", source_lang='de', target_lang='en'):
    """
    Translate text from source language to target language using OpenAI's API.
    Default is German to English translation.
    
    Handles API limitations by chunking text
    """
    if not text or len(text.strip()) == 0:
        return text
    
    # Log translation attempt
    logger.info(f"Translating text ({len(text)} characters) using {model}")
    
    # Maximum size for a chunk to send to OpenAI
    max_chunk_size = 4000  # Conservative limit for context window
    
    # Split text into paragraphs
    paragraphs = text.split('\n')
    all_translated_chunks = []
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the chunk size, translate the current chunk first
        if len(current_chunk) + len(paragraph) + 1 > max_chunk_size:
            if current_chunk:
                try:
                    # Create the prompt for translation
                    messages = [
                        {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}. Translate the text exactly as provided, maintaining the original meaning and tone. Respond only with the translation, adding no comments or explanations."},
                        {"role": "user", "content": current_chunk}
                    ]
                    
                    # Call the OpenAI API
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3  # Lower temperature for more accurate translations
                    )
                    
                    # Extract the translated text
                    translation = response.choices[0].message.content.strip()
                    all_translated_chunks.append(translation)
                    
                    # Log successful translation
                    logger.info(f"Successfully translated chunk ({len(current_chunk)} chars)")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Translation error: {str(e)}")
                    # Keep original text if translation fails
                    all_translated_chunks.append(f"[TRANSLATION ERROR: {str(e)}]\n{current_chunk}")
                
                current_chunk = paragraph
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Translate any remaining text
    if current_chunk:
        try:
            # Create the prompt for translation
            messages = [
                {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}. Translate the text exactly as provided, maintaining the original meaning and tone. Respond only with the translation, adding no comments or explanations."},
                {"role": "user", "content": current_chunk}
            ]
            
            # Call the OpenAI API
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3
            )
            
            # Extract the translated text
            translation = response.choices[0].message.content.strip()
            all_translated_chunks.append(translation)
            
            # Log successful translation
            logger.info(f"Successfully translated final chunk ({len(current_chunk)} chars)")
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            all_translated_chunks.append(f"[TRANSLATION ERROR: {str(e)}]\n{current_chunk}")
    
    # Combine translated chunks
    result = '\n'.join(all_translated_chunks)
    
    # Log translation completion
    logger.info(f"Translation complete. Original: {len(text)} chars, Translated: {len(result)} chars")
    
    return result

def is_german_document(pdf_path, sample_size=20, german_threshold=0.3, min_confidence=0.6):
    """
    Determines if a PDF document is predominantly in German.
    
    Args:
        pdf_path: Path to the PDF file
        sample_size: Number of text samples to take
        german_threshold: Threshold to consider the document as German
        min_confidence: Minimum confidence for language detection
        
    Returns:
        tuple: (is_german, percentage_german, confidence_avg)
    """
    try:
        # Try to safely open the PDF file
        try:
            reader = PdfReader(pdf_path, strict=False)
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PyPDF2 could not read {pdf_path}: {str(e)}")
            return False, 0.0, 0.0
        
        # Get total pages
        try:
            total_pages = len(reader.pages)
            if total_pages == 0:
                logger.warning(f"PDF {pdf_path} has no pages")
                return False, 0.0, 0.0
        except Exception as e:
            logger.error(f"Could not get page count for {pdf_path}: {str(e)}")
            return False, 0.0, 0.0
        
        # Calculate pages to sample
        pages_to_sample = min(sample_size, total_pages)
        
        # Calculate step size to distribute samples across document
        step = max(1, total_pages // pages_to_sample)
        
        # Extract all text to check if document is empty or unreadable
        all_text = ""
        for i in range(min(5, total_pages)):  # Check first 5 pages at most
            try:
                page_text = reader.pages[i].extract_text()
                if page_text:
                    all_text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {i} of {pdf_path}: {str(e)}")
                
        # If we couldn't extract any text, try to be more lenient
        if not all_text.strip():
            logger.warning(f"Could not extract any text from first pages of {pdf_path}")
            
            # Try harder with a different approach - check more pages
            for i in range(total_pages):
                try:
                    page_text = reader.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 100:  # Found a page with significant text
                        all_text = page_text
                        break
                except Exception:
                    continue
                    
            if not all_text.strip():
                logger.error(f"Document {pdf_path} appears to have no extractable text")
                return False, 0.0, 0.0
        
        # Quick check: look for common German words in the text
        german_markers = ["und", "der", "die", "das", "mit", "für", "ist", "sind", "werden", "wurde"]
        german_marker_count = sum(1 for marker in german_markers if f" {marker} " in all_text.lower())
        
        # If we have strong indicators of German text, don't need further analysis
        if german_marker_count >= 5:
            logger.info(f"Document {pdf_path} contains multiple German markers ({german_marker_count})")
            return True, 0.9, 0.9
            
        german_count = 0
        confidence_sum = 0
        valid_samples = 0
        errors = 0
        max_errors = min(5, total_pages // 2)  # Allow some errors, but not too many
        
        # Extract text from sample pages
        for i in range(0, total_pages, step):
            if valid_samples >= sample_size:
                break
                
            if i >= total_pages:
                continue
            
            try:
                page = reader.pages[i]
                text = page.extract_text()
                
                # Skip pages with little or no text
                if not text or len(text.strip()) < 50:
                    continue
                    
                lang, confidence = detect_language(text, min_confidence)
                
                # Only count samples with sufficient confidence
                if confidence >= min_confidence:
                    valid_samples += 1
                    confidence_sum += confidence
                    if lang == 'de':
                        german_count += 1
            except Exception as e:
                errors += 1
                logger.warning(f"Could not process page {i} of {pdf_path}: {str(e)}")
                if errors > max_errors:
                    logger.error(f"Too many errors ({errors}) processing {pdf_path}. Aborting.")
                    return False, 0.0, 0.0
        
        # If we couldn't get enough valid samples, try a different approach
        if valid_samples < 3:
            # Last resort: check the entire document's text at once
            lang, confidence = detect_language(all_text, min_confidence - 0.1)  # Be more lenient with confidence
            if lang == 'de' and confidence >= min_confidence - 0.1:
                logger.info(f"Document {pdf_path} identified as German using full-text approach")
                return True, 0.8, confidence
                
            logger.warning(f"Not enough valid text samples in {pdf_path} (found {valid_samples})")
            return False, 0.0, 0.0
            
        # Calculate percentage of German pages
        percent_german = german_count / valid_samples if valid_samples > 0 else 0
        avg_confidence = confidence_sum / valid_samples if valid_samples > 0 else 0
        
        logger.info(f"Language analysis for {pdf_path}: {valid_samples} valid samples, {percent_german:.2f} German, {avg_confidence:.2f} confidence")
        return percent_german >= german_threshold, percent_german, avg_confidence
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False, 0.0, 0.0

def translate_pdf(input_path, output_path, model="gpt-4o-mini"):
    """
    Translate a PDF from German to English using OpenAI and save to output path.
    
    This creates a new PDF with the translated text content.
    """
    try:
        # Open the PDF file with error handling
        try:
            reader = PdfReader(input_path, strict=False)
            writer = PdfWriter()
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PyPDF2 could not read {input_path}: {str(e)}")
            # Create a text file explaining the issue
            text_output_path = os.path.splitext(output_path)[0] + "_error.txt"
            with open(text_output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(f"ERROR: Could not process {input_path}\n\n")
                text_file.write(f"PyPDF2 error: {str(e)}\n\n")
                text_file.write("Try opening this PDF with Adobe Acrobat or another PDF reader and resaving it.")
            return False
        
        # Extract all text first to optimize API calls
        all_pages_text = []
        extractable_pages = []
        error_pages = []
        
        logger.info(f"Extracting text from {input_path} with {len(reader.pages)} pages")
        
        for i, page in enumerate(tqdm(reader.pages, desc="Extracting text")):
            try:
                text = page.extract_text()
                
                # Skip translation if page has no text
                if not text or len(text.strip()) < 10:
                    all_pages_text.append(None)
                    extractable_pages.append(False)
                    error_pages.append(False)
                else:
                    all_pages_text.append(text)
                    extractable_pages.append(True)
                    error_pages.append(False)
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                all_pages_text.append(None)
                extractable_pages.append(False)
                error_pages.append(True)
        
        # Count extractable pages
        total_extractable = sum(extractable_pages)
        if total_extractable == 0:
            logger.error(f"No extractable text found in {input_path}")
            # Create a text file explaining the issue
            text_output_path = os.path.splitext(output_path)[0] + "_error.txt"
            with open(text_output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(f"ERROR: No extractable text found in {input_path}\n\n")
                text_file.write("This PDF may be scanned without OCR or contain only images.\n")
                text_file.write("Consider running OCR software before translation.")
            
            # Still copy the file to output with appropriate suffix
            shutil.copy2(input_path, output_path)
            return False
        
        # Now translate all pages with text
        translated_texts = []
        translation_errors = 0
        
        logger.info(f"Translating {total_extractable} pages with text")
        
        for i, text in enumerate(tqdm([t for t in all_pages_text if t], desc="Translating pages")):
            try:
                translated = translate_text_with_openai(text, model=model)
                translated_texts.append(translated)
                
                # Avoid rate limiting
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Translation error on page {i+1}: {str(e)}")
                translated_texts.append(f"[Translation error: {str(e)}]")
                translation_errors += 1
        
        # If too many translation errors, consider it a failure
        if translation_errors > total_extractable / 2:
            logger.error(f"Too many translation errors ({translation_errors}/{total_extractable}) for {input_path}")
            return False
        
        # Create the output PDF (copy original)
        try:
            # Copy the original PDF to the output path first
            shutil.copy2(input_path, output_path)
            
            # Then create the text file with translations
            text_output_path = os.path.splitext(output_path)[0] + "_translation.txt"
            
            with open(text_output_path, 'w', encoding='utf-8') as text_file:
                text_file.write("# TRANSLATED CONTENT\n\n")
                text_file.write(f"Original file: {os.path.basename(input_path)}\n")
                text_file.write(f"Translation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                text_file.write(f"Translation model: {model}\n\n")
                
                translation_idx = 0
                for i, page_has_text in enumerate(extractable_pages):
                    text_file.write(f"## Page {i+1}\n\n")
                    
                    if error_pages[i]:
                        text_file.write("[Error extracting text from this page]\n\n")
                    elif page_has_text:
                        text_file.write(translated_texts[translation_idx])
                        translation_idx += 1
                        text_file.write("\n\n")
                    else:
                        text_file.write("[No extractable text on this page]\n\n")
            
            logger.info(f"Successfully created translation file: {text_output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating output files: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error translating PDF {input_path}: {str(e)}")
        return False

def already_translated(filename, output_dir):
    """
    Check if a file has already been translated.
    More thorough check that looks for various translated file patterns.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Check for different possible translated file patterns
    translation_patterns = [
        f"{base_name}_EN.pdf",                     # Standard English suffix
        f"{base_name}_translation.txt",            # Translation text file
        f"{base_name}_EN_translation.txt",         # Another possible pattern
        f"{base_name}-EN.pdf",                     # Hyphen variant
        f"{base_name}.en.pdf",                     # Dot variant
        f"{base_name}_translated.pdf",             # Alternative naming
        f"{base_name}_translated_to_english.pdf",  # Verbose naming
    ]
    
    # Check if any of these files exist
    for pattern in translation_patterns:
        if os.path.exists(os.path.join(output_dir, pattern)):
            logger.info(f"Found existing translation for {filename}: {pattern}")
            return True
    
    # Also check for the translation text file with or without PDF
    translation_text_file = os.path.join(output_dir, f"{base_name}_translation.txt")
    if os.path.exists(translation_text_file):
        # Read the first few lines to verify it's a real translation
        try:
            with open(translation_text_file, 'r', encoding='utf-8') as f:
                first_lines = ''.join([f.readline() for _ in range(10)]).lower()
                if 'translation' in first_lines or 'translated' in first_lines:
                    logger.info(f"Found existing translation text file for {filename}")
                    return True
        except Exception:
            pass  # If we can't read the file, assume it's not a valid translation
            
    # If we've already processed this file, check our records
    translation_registry = os.path.join(output_dir, "translated_files_registry.txt")
    if os.path.exists(translation_registry):
        try:
            with open(translation_registry, 'r', encoding='utf-8') as f:
                registry_content = f.read()
                # Check if this file is in the registry
                if os.path.basename(filename) in registry_content:
                    logger.info(f"File {filename} found in translation registry")
                    return True
        except Exception as e:
            logger.warning(f"Error checking translation registry: {str(e)}")
    
    return False

def register_translated_file(filename, output_dir, success=True):
    """
    Register a file in the translation registry.
    This helps track which files have already been processed.
    """
    translation_registry = os.path.join(output_dir, "translated_files_registry.txt")
    
    # Create registry if it doesn't exist
    if not os.path.exists(translation_registry):
        try:
            with open(translation_registry, 'w', encoding='utf-8') as f:
                f.write(f"Translation Registry - Created on {datetime.now().isoformat()}\n")
                f.write("Format: filename,status,date\n\n")
        except Exception as e:
            logger.error(f"Error creating translation registry: {str(e)}")
            return False
    
    # Add entry to registry
    try:
        with open(translation_registry, 'a', encoding='utf-8') as f:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{os.path.basename(filename)},{status},{datetime.now().isoformat()}\n")
        return True
    except Exception as e:
        logger.error(f"Error updating translation registry: {str(e)}")
        return False

def process_directory(input_dir, output_dir, args):
    """
    Process all PDF files in the input directory, detect German documents,
    and translate them to the output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Lists to track problematic files
    error_files = []
    
    # Process each PDF file
    results = {
        'total': len(pdf_files),
        'german': 0,
        'english': 0,
        'other': 0,
        'translated': 0,
        'already_translated': 0,
        'failed': 0,
        'skipped': 0,
        'error': 0
    }
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_EN.pdf")
            
            # Skip if already translated and not forcing translation
            if already_translated(pdf_file, output_dir) and not args.force_translate:
                logger.info(f"Skipping already translated file: {pdf_file}")
                results['already_translated'] += 1
                continue
            
            # Try to check if the file is in German
            try:
                is_german, percent_german, confidence = is_german_document(
                    pdf_file, 
                    sample_size=args.sample_size,
                    german_threshold=args.german_threshold,
                    min_confidence=args.min_confidence
                )
            except Exception as e:
                logger.error(f"Error analyzing language in {pdf_file}: {str(e)}")
                error_files.append((pdf_file, f"Language detection error: {str(e)}"))
                results['error'] += 1
                
                # Copy the file to output with ERROR suffix
                error_copy = os.path.join(output_dir, f"{base_name}_ERROR.pdf")
                try:
                    shutil.copy2(pdf_file, error_copy)
                except Exception as copy_err:
                    logger.error(f"Could not copy file {pdf_file}: {str(copy_err)}")
                
                continue
            
            if is_german:
                logger.info(f"Detected German document: {pdf_file} (German: {percent_german:.2f}, Confidence: {confidence:.2f})")
                results['german'] += 1
                
                if not args.no_translate:
                    # Try to translate the PDF
                    try:
                        logger.info(f"Translating: {pdf_file} -> {output_file}")
                        success = translate_pdf(pdf_file, output_file, model=args.openai_model)
                        
                        # Update results and register
                        if success:
                            results['translated'] += 1
                            register_translated_file(pdf_file, output_dir, success=True)
                        else:
                            results['failed'] += 1
                            register_translated_file(pdf_file, output_dir, success=False)
                            error_files.append((pdf_file, "Translation failed"))
                    except Exception as e:
                        logger.error(f"Translation error for {pdf_file}: {str(e)}")
                        error_files.append((pdf_file, f"Translation error: {str(e)}"))
                        results['failed'] += 1
                        
                        # Copy the file to output with ERROR suffix
                        error_copy = os.path.join(output_dir, f"{base_name}_ERROR.pdf")
                        try:
                            shutil.copy2(pdf_file, error_copy)
                        except Exception as copy_err:
                            logger.error(f"Could not copy file {pdf_file}: {str(copy_err)}")
                else:
                    # Just copy the file with a different name to mark it as German
                    german_copy = os.path.join(output_dir, f"{base_name}_DE.pdf")
                    try:
                        shutil.copy2(pdf_file, german_copy)
                        results['skipped'] += 1
                    except Exception as copy_err:
                        logger.error(f"Could not copy file {pdf_file}: {str(copy_err)}")
                        error_files.append((pdf_file, f"File copy error: {str(copy_err)}"))
                        results['error'] += 1
            else:
                # Check if it's likely English or another language
                try:
                    try:
                        reader = PdfReader(pdf_file, strict=False)
                        if len(reader.pages) > 0:
                            sample_text = reader.pages[0].extract_text()
                            if sample_text:
                                lang, _ = detect_language(sample_text)
                                if lang == 'en':
                                    results['english'] += 1
                                    # Copy English files to output directory with _EN suffix
                                    english_copy = os.path.join(output_dir, f"{base_name}_EN.pdf")
                                    shutil.copy2(pdf_file, english_copy)
                                else:
                                    results['other'] += 1
                                    # Copy other language files with appropriate suffix
                                    other_copy = os.path.join(output_dir, f"{base_name}_{lang.upper()}.pdf")
                                    shutil.copy2(pdf_file, other_copy)
                            else:
                                # No text on first page, mark as unknown
                                results['other'] += 1
                                unknown_copy = os.path.join(output_dir, f"{base_name}_UNKNOWN.pdf")
                                shutil.copy2(pdf_file, unknown_copy)
                        else:
                            # No pages, mark as error
                            results['error'] += 1
                            error_files.append((pdf_file, "PDF has no pages"))
                            error_copy = os.path.join(output_dir, f"{base_name}_ERROR.pdf")
                            shutil.copy2(pdf_file, error_copy)
                    except Exception as pdf_err:
                        # Error reading PDF, still copy it
                        logger.error(f"Error reading {pdf_file}: {str(pdf_err)}")
                        error_files.append((pdf_file, f"PDF read error: {str(pdf_err)}"))
                        results['error'] += 1
                        error_copy = os.path.join(output_dir, f"{base_name}_ERROR.pdf")
                        shutil.copy2(pdf_file, error_copy)
                except Exception as e:
                    logger.error(f"Unexpected error processing {pdf_file}: {str(e)}")
                    error_files.append((pdf_file, f"Unexpected error: {str(e)}"))
                    results['error'] += 1
        except Exception as e:
            # Catch any other errors to ensure processing continues
            logger.error(f"Critical error while processing {pdf_file}: {str(e)}")
            logger.error(traceback.format_exc())
            error_files.append((pdf_file, f"Critical error: {str(e)}"))
            results['error'] += 1
    
    # Write error files to a log
    if error_files:
        error_log_path = os.path.join(output_dir, "problematic_pdfs.txt")
        with open(error_log_path, 'w', encoding='utf-8') as error_log:
            error_log.write(f"Problematic PDF Files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            error_log.write(f"Total problematic files: {len(error_files)}\n\n")
            
            for file_path, error_msg in error_files:
                error_log.write(f"File: {file_path}\n")
                error_log.write(f"Error: {error_msg}\n")
                error_log.write("-" * 80 + "\n\n")
        
        logger.info(f"Wrote list of {len(error_files)} problematic PDFs to {error_log_path}")
    
    return results

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    logger.info("Starting PDF Language Detection and Translation")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"OpenAI model: {args.openai_model}")
    
    # Set up OpenAI API
    if not setup_openai_api(args.openai_api_key):
        logger.error("Failed to set up OpenAI API. Exiting.")
        exit(1)
    
    start_time = time.time()
    results = process_directory(args.input_dir, args.output_dir, args)
    end_time = time.time()
    
    # Log results
    logger.info("\nProcessing Complete!")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total PDFs: {results['total']}")
    logger.info(f"German PDFs: {results['german']}")
    logger.info(f"English PDFs: {results['english']}")
    logger.info(f"Other language PDFs: {results['other']}")
    logger.info(f"Problematic PDFs: {results['error']}")
    
    if not args.no_translate:
        logger.info(f"Successfully translated: {results['translated']}")
        logger.info(f"Already translated (skipped): {results['already_translated']}")
        logger.info(f"Failed translations: {results['failed']}")
    else:
        logger.info(f"Translation skipped (detection only mode)")
    
    # Check if there were any problematic files
    error_log_path = os.path.join(output_dir, "problematic_pdfs.txt")
    if os.path.exists(error_log_path):
        logger.info(f"Some PDFs had errors. See {error_log_path} for details.")