import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from tqdm import tqdm
import argparse
from urllib.robotparser import RobotFileParser
import random
import re
from collections import defaultdict

def is_valid_url(url):
    """Check if url is a valid URL."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def is_pdf_link(url):
    """Check if the URL points to a PDF file."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith('.pdf')

def get_robots_parser(url, ignore_robots=False):
    """Create and return a RobotFileParser for the given URL."""
    # If ignore_robots is True, return a parser that allows everything
    if ignore_robots:
        parser = RobotFileParser()
        # Fake an empty robots.txt that allows everything
        parser.parse([''])
        return parser
    
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    parser = RobotFileParser()
    try:
        parser.set_url(robots_url)
        parser.read()
        return parser
    except Exception as e:
        print(f"Error reading robots.txt from {robots_url}: {e}")
        # Return a parser that allows everything
        parser = RobotFileParser()
        parser.parse([''])
        return parser

def get_all_links(url, robots_parser, user_agent="python-pdf-scraper", ignore_robots=False):
    """Extract all links from a webpage, respecting robots.txt unless ignored."""
    try:
        # Check if we're allowed to fetch this URL, bypass if ignore_robots is True
        if not ignore_robots and not robots_parser.can_fetch(user_agent, url):
            print(f"Robots.txt disallows access to {url}")
            if ignore_robots:
                print("...but proceeding anyway as robots.txt is being ignored")
            else:
                return []
        
        # Add a random delay to be nice to the server (between 1 and 3 seconds)
        time.sleep(random.uniform(1, 3))
        
        headers = {
            'User-Agent': user_agent
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            if link and not link.startswith('#'):  # Ignore anchors
                # Handle relative URLs
                absolute_link = urljoin(url, link)
                if is_valid_url(absolute_link):
                    links.append(absolute_link)
        
        return links
    except requests.exceptions.RequestException as e:
        print(f"Network error for {url}: {e}")
        return []
    except Exception as e:
        print(f"Error extracting links from {url}: {e}")
        return []

def download_pdf(pdf_url, download_folder, robots_parser, user_agent="python-pdf-scraper", ignore_robots=False):
    """Download a PDF file from the given URL."""
    try:
        # Check if we're allowed to fetch this URL
        if not ignore_robots and not robots_parser.can_fetch(user_agent, pdf_url):
            print(f"Robots.txt disallows download of {pdf_url}")
            if ignore_robots:
                print("...but proceeding anyway as robots.txt is being ignored")
            else:
                return False
        
        # Extract filename from URL
        parsed_url = urlparse(pdf_url)
        filename = os.path.basename(parsed_url.path)
        
        # If filename is empty or invalid, generate a name based on URL hash
        if not filename or '.' not in filename:
            filename = f"document_{hash(pdf_url) % 10000}.pdf"
        
        # Create the download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)
        
        # Full path for the downloaded file
        filepath = os.path.join(download_folder, filename)
        
        # Check if file already exists, generate a new name if it does
        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                new_filename = f"{base}_{counter}{ext}"
                filepath = os.path.join(download_folder, new_filename)
                counter += 1
            filename = os.path.basename(filepath)
            print(f"File already exists, renaming to {filename}")
        
        # Add a small delay to be nice to the server
        time.sleep(random.uniform(0.5, 1.5))
        
        headers = {
            'User-Agent': user_agent
        }
        
        # Stream the download with progress bar
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Verify it's a PDF by checking Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            print(f"Warning: {pdf_url} may not be a PDF (Content-Type: {content_type})")
        
        # Get the file size from headers (if available)
        total_size = int(response.headers.get('content-length', 0))
        
        # Create a progress bar
        desc = f"Downloading {filename}"
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Network error downloading {pdf_url}: {e}")
        return False
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False

def scrape_website_for_pdfs(url, download_folder, recursive=False, max_depth=1, max_pdfs=None, ignore_robots=False):
    """
    Scrape a website for PDF files.
    
    Args:
        url (str): The URL of the website to scrape.
        download_folder (str): The folder where PDFs will be saved.
        recursive (bool): Whether to follow links recursively.
        max_depth (int): Maximum depth for recursive scraping.
        max_pdfs (int, optional): Maximum number of PDFs to download.
        ignore_robots (bool): Whether to ignore robots.txt restrictions.
    """
    print(f"Scraping {url} for PDF files...")
    if ignore_robots:
        print("Note: robots.txt restrictions will be ignored.")
    
    # Initialize robots.txt parser
    robots_parser = get_robots_parser(url, ignore_robots)
    user_agent = "python-pdf-scraper"
    
    # Keep track of visited URLs to avoid loops
    visited_urls = set()
    # Queue of URLs to visit with their depth
    url_queue = [(url, 0)]
    # Track the number of PDFs found and downloaded
    pdfs_found = 0
    pdfs_downloaded = 0
    
    while url_queue:
        current_url, depth = url_queue.pop(0)
        
        # Skip if we've already visited this URL
        if current_url in visited_urls:
            continue
        
        # Check if we've reached the maximum number of PDFs
        if max_pdfs is not None and pdfs_downloaded >= max_pdfs:
            print(f"Reached maximum number of PDFs to download ({max_pdfs})")
            break
        
        visited_urls.add(current_url)
        print(f"Checking page: {current_url}")
        
        # Get all links from the current page
        links = get_all_links(current_url, robots_parser, user_agent, ignore_robots)
        
        # Process each link
        for link in links:
            # Check if we've reached the maximum number of PDFs
            if max_pdfs is not None and pdfs_downloaded >= max_pdfs:
                break
                
            if is_pdf_link(link):
                pdfs_found += 1
                print(f"Found PDF: {link}")
                if download_pdf(link, download_folder, robots_parser, user_agent, ignore_robots):
                    pdfs_downloaded += 1
            elif recursive and depth < max_depth and link not in visited_urls:
                # Only add to the queue if we're doing recursive scraping
                # and haven't reached the maximum depth
                url_queue.append((link, depth + 1))
    
    print(f"\nScraping complete!")
    print(f"Found {pdfs_found} PDF files")
    print(f"Successfully downloaded {pdfs_downloaded} PDF files to {download_folder}")

def main():
    parser = argparse.ArgumentParser(description='Scrape a website for PDF files.')
    parser.add_argument('url', help='The URL of the website to scrape')
    parser.add_argument('--output', '-o', default='./downloaded_pdfs', 
                        help='Output directory for downloaded PDFs (default: ./downloaded_pdfs)')
    parser.add_argument('--recursive', '-r', action='store_true', 
                        help='Recursively follow links to find PDFs')
    parser.add_argument('--depth', '-d', type=int, default=1, 
                        help='Maximum depth for recursive scraping (default: 1)')
    parser.add_argument('--max-pdfs', '-m', type=int, default=None,
                        help='Maximum number of PDFs to download (default: no limit)')
    parser.add_argument('--user-agent', '-u', default='python-pdf-scraper',
                        help='User agent string to use for requests (default: python-pdf-scraper)')
    parser.add_argument('--ignore-robots', '-i', action='store_true',
                        help='Ignore robots.txt restrictions (use responsibly)')
    
    args = parser.parse_args()
    
    scrape_website_for_pdfs(args.url, args.output, args.recursive, args.depth, args.max_pdfs, args.ignore_robots)

if __name__ == "__main__":
    main()
