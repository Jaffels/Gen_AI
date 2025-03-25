import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
from urllib.robotparser import RobotFileParser
import time
import random
import argparse
import logging
from tqdm import tqdm


class PDFScraper:
    def __init__(self, base_url, output_dir="downloaded_pdfs", delay=1, max_depth=3):
        """
        Initialize the PDF scraper.

        Args:
            base_url (str): The starting URL to scrape
            output_dir (str): Directory to save PDFs
            delay (float): Seconds to wait between requests
            max_depth (int): Maximum depth for recursive crawling
        """
        self.base_url = base_url
        self.base_domain = urllib.parse.urlparse(base_url).netloc
        self.output_dir = output_dir
        self.delay = delay
        self.max_depth = max_depth

        # Setup tracking sets
        self.visited_urls = set()
        self.pdf_urls = set()

        # Setup logging
        self.setup_logging()

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Setup robots.txt parser
        self.rp = RobotFileParser()
        robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
        try:
            self.rp.set_url(robots_url)
            self.rp.read()

            # Get crawl-delay from robots.txt if available
            crawl_delay = self.rp.crawl_delay("*")
            if crawl_delay:
                self.delay = max(self.delay, crawl_delay)
                self.logger.info(f"Setting crawl delay to {self.delay}s based on robots.txt")
        except Exception as e:
            self.logger.warning(f"Could not read robots.txt: {e}")

    def setup_logging(self):
        """Configure logging for the scraper"""
        self.logger = logging.getLogger("PDFScraper")
        self.logger.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def is_allowed(self, url):
        """Check if the URL is allowed by robots.txt"""
        try:
            return self.rp.can_fetch("*", url)
        except Exception:
            # If we can't check robots.txt, assume it's allowed
            return True

    def normalize_url(self, url, current_url):
        """Convert relative URLs to absolute and normalize them"""
        # Skip fragments, javascript, mailto links
        if url.startswith(('#', 'javascript:', 'mailto:')):
            return None

        # Convert relative URLs to absolute
        if not url.startswith(('http://', 'https://')):
            url = urllib.parse.urljoin(current_url, url)

        # Parse URL
        parsed = urllib.parse.urlparse(url)

        # Basic normalization
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path

        # Remove trailing slashes
        if path.endswith('/') and len(path) > 1:
            path = path[:-1]

        # Rebuild URL without fragment
        normalized = urllib.parse.urlunparse((
            scheme, netloc, path, parsed.params, parsed.query, ""
        ))

        return normalized

    def is_same_domain(self, url):
        """Check if the URL is from the same domain as the base URL"""
        domain = urllib.parse.urlparse(url).netloc
        return domain == self.base_domain or domain == ''

    def is_pdf_link(self, url):
        """Check if the URL points to a PDF file"""
        # Check file extension
        if url.lower().endswith('.pdf'):
            return True

        # Check URL for indicators
        url_lower = url.lower()
        if "pdf" in url_lower or "document" in url_lower:
            try:
                # Check content type with HEAD request
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('Content-Type', '').lower()

                if 'application/pdf' in content_type:
                    return True

                # Check Content-Disposition header
                disposition = response.headers.get('Content-Disposition', '')
                if 'attachment' in disposition and '.pdf' in disposition:
                    return True
            except Exception as e:
                self.logger.debug(f"Error checking PDF: {e}")

        return False

    def download_pdf(self, url):
        """Download a PDF file from the given URL"""
        # Extract or generate filename
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If no filename or doesn't end with .pdf, generate one
        if not filename or not filename.lower().endswith('.pdf'):
            filename = f"document_{len(self.pdf_urls)}.pdf"

        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
        if not filename:
            filename = f"document_{len(self.pdf_urls)}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        # Handle duplicate filenames
        counter = 1
        original_filename = filename
        while os.path.exists(filepath):
            name_parts = os.path.splitext(original_filename)
            filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
            filepath = os.path.join(self.output_dir, filename)
            counter += 1

        try:
            # Download the file
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                self.logger.warning(f"URL {url} does not appear to be a PDF (Content-Type: {content_type})")
                return False

            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(filepath, 'wb') as file, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    size = file.write(chunk)
                    bar.update(size)

            self.logger.info(f"Downloaded: {filename} ({total_size} bytes)")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            # Remove partial downloads
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def scrape_page(self, url, depth=0):
        """Scrape a page for PDF links and subpages"""
        # Check depth limit
        if depth > self.max_depth:
            self.logger.debug(f"Reached maximum depth ({self.max_depth}) at {url}")
            return

        # Normalize URL
        url = self.normalize_url(url, url)
        if not url:
            return

        # Skip if already visited or not allowed
        if url in self.visited_urls:
            return

        if not self.is_allowed(url):
            self.logger.debug(f"Skipping {url} - not allowed by robots.txt")
            return

        # Mark as visited to avoid loops
        self.visited_urls.add(url)

        self.logger.info(f"Scraping: {url} (depth: {depth}/{self.max_depth})")

        try:
            # Add random delay to avoid overloading the server
            time.sleep(self.delay * (0.8 + 0.4 * random.random()))

            # Make the request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract all links
            links = soup.find_all('a', href=True)
            pdf_links = []
            page_links = []

            # Process links
            for link in links:
                href = link['href']
                full_url = self.normalize_url(href, url)

                if not full_url:
                    continue

                # Categorize the link
                if self.is_pdf_link(full_url):
                    if full_url not in self.pdf_urls:
                        pdf_links.append(full_url)
                elif self.is_same_domain(full_url) and full_url not in self.visited_urls:
                    page_links.append(full_url)

            # Download PDFs
            for pdf_url in pdf_links:
                self.pdf_urls.add(pdf_url)
                self.download_pdf(pdf_url)

            # Recursively scrape subpages
            for page_url in page_links:
                self.scrape_page(page_url, depth + 1)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {url}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")

    def start(self):
        """Start the scraping process"""
        self.logger.info(f"Starting PDF scraper at: {self.base_url}")
        self.logger.info(f"PDFs will be saved to: {os.path.abspath(self.output_dir)}")
        self.logger.info(f"Maximum crawl depth: {self.max_depth}")

        try:
            start_time = time.time()
            self.scrape_page(self.base_url)
            end_time = time.time()

            duration = end_time - start_time
            self.logger.info(f"Scraping completed in {duration:.2f} seconds")
            self.logger.info(f"Found {len(self.pdf_urls)} PDF files")
            self.logger.info(f"Visited {len(self.visited_urls)} pages")

            # Save list of downloaded PDFs
            if self.pdf_urls:
                pdf_list_file = os.path.join(self.output_dir, "downloaded_pdfs.txt")
                with open(pdf_list_file, "w") as f:
                    for pdf_url in sorted(self.pdf_urls):
                        f.write(f"{pdf_url}\n")
                self.logger.info(f"Saved list of downloaded PDFs to {pdf_list_file}")

        except KeyboardInterrupt:
            self.logger.info("Scraping interrupted by user")
            self.logger.info(f"Found {len(self.pdf_urls)} PDF files before interruption")
        except Exception as e:
            self.logger.error(f"Scraping failed with error: {str(e)}")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Download PDFs from a website and its subpages.')
    parser.add_argument('url', type=str, help='The base URL to start scraping from')
    parser.add_argument('--output', '-o', type=str, default='downloaded_pdfs',
                        help='Directory to save downloaded PDFs (default: downloaded_pdfs)')
    parser.add_argument('--delay', '-d', type=float, default=1,
                        help='Delay between requests in seconds (default: 1)')
    parser.add_argument('--max-depth', '-m', type=int, default=3,
                        help='Maximum crawl depth (default: 3)')
    args = parser.parse_args()

    # Create and run the scraper
    scraper = PDFScraper(
        base_url=args.url,
        output_dir=args.output,
        delay=args.delay,
        max_depth=args.max_depth
    )

    try:
        scraper.start()
    except KeyboardInterrupt:
        print("\nScraping interrupted. Exiting gracefully...")


if __name__ == '__main__':
    main()