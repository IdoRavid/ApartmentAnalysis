import time
import random
import re
import os
import json
import urllib.parse
import shutil
import subprocess
import sys
import datetime
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import chromedriver_autoinstaller
import tempfile


class FacebookPostExtractor:
    def __init__(self, chrome_user_data_dir=None, profile_directory="Default", use_profile_copy=True):
        # Automatically installs the correct version of ChromeDriver
        chromedriver_autoinstaller.install()

        # Set up Chrome options
        self.chrome_options = Options()

        # Add these arguments to prevent the DevToolsActivePort error
        self.chrome_options.add_argument("--remote-debugging-port=9223")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")

        # Handle profile copying if requested
        self.original_profile = chrome_user_data_dir
        self.profile_copy_dir = os.path.join(os.getcwd(), "chrome_profile_copy")

        # If using profile copy, set it up
        if use_profile_copy and self.original_profile:
            self._setup_profile_copy()
            self.chrome_options.add_argument(f"user-data-dir={self.profile_copy_dir}")
        elif chrome_user_data_dir:
            # Use the original profile directly if not copying
            self.chrome_options.add_argument(f"user-data-dir={chrome_user_data_dir}")

        if profile_directory:
            self.chrome_options.add_argument(f"profile-directory={profile_directory}")

        # Common browser options
        self.chrome_options.add_argument('--disable-extensions')
        self.chrome_options.add_argument('--disable-gpu')

        # Randomize User-Agent to avoid bot detection
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"
        ]
        self.chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")

        # Initialize driver as None, will be created when needed
        self.driver = None
        self.unique_posts = set()

    def _setup_profile_copy(self, force_new=False):
        """Set up a copy of the Chrome profile to avoid DevToolsActivePort issues"""
        # Check if we need to update the profile copy
        profile_info_file = os.path.join(self.profile_copy_dir, "profile_info.txt")

        # Delete existing profile copy if force_new is True
        if force_new and os.path.exists(self.profile_copy_dir):
            try:
                shutil.rmtree(self.profile_copy_dir)
                print("Deleted existing profile copy to create a fresh one.")
            except Exception as e:
                print(f"Warning: Could not delete old profile copy: {e}")

        # Check if we already have a recent copy
        if os.path.exists(profile_info_file) and not force_new:
            with open(profile_info_file, 'r') as f:
                creation_date = f.read().strip()
                try:
                    creation_date = datetime.strptime(creation_date, "%Y-%m-%d %H:%M:%S")
                    age_days = (datetime.now() - creation_date).days
                    print(f"Using existing profile copy created {age_days} days ago.")
                    if age_days > 7:  # Profile copy is older than 7 days
                        choice = input("Profile copy is older than 7 days. Create a fresh copy? (y/n): ")
                        if choice.lower() == 'y':
                            return self._setup_profile_copy(force_new=True)
                    return
                except:
                    # If we can't parse the date, create a new copy
                    pass

        # First ensure Chrome is not running
        try:
            if os.name == 'nt':  # Windows
                output = subprocess.check_output('tasklist /FI "IMAGENAME eq chrome.exe"', shell=True).decode()
                if "chrome.exe" in output:
                    print("Chrome is running. Please close ALL Chrome windows before creating a profile copy.")
                    choice = input("Continue anyway? (y/n): ")
                    if choice.lower() != 'y':
                        sys.exit(0)
            else:  # Linux/Mac
                try:
                    output = subprocess.check_output(['pgrep', 'chrome']).decode()
                    if output.strip():
                        print("Chrome is running. Please close ALL Chrome windows before creating a profile copy.")
                        choice = input("Continue anyway? (y/n): ")
                        if choice.lower() != 'y':
                            sys.exit(0)
                except subprocess.CalledProcessError:
                    # Chrome is not running, which is good
                    pass
        except Exception as e:
            print(f"Could not check if Chrome is running: {e}")
            print("Please ensure all Chrome windows are closed before continuing.")

        # Create the profile copy directory
        os.makedirs(self.profile_copy_dir, exist_ok=True)

        print("Creating a copy of your Chrome profile. This may take a few moments...")

        try:
            # Copy essential folders and files for maintaining login sessions
            essential_items = ['Default', 'Local State']

            for item in essential_items:
                source_path = os.path.join(self.original_profile, item)
                dest_path = os.path.join(self.profile_copy_dir, item)

                if os.path.exists(source_path):
                    if os.path.isdir(source_path):
                        # If it's a directory and already exists, don't copy again to avoid permission issues
                        if not os.path.exists(dest_path):
                            print(f"Copying {item} folder...")
                            shutil.copytree(source_path, dest_path)
                    else:
                        print(f"Copying {item} file...")
                        shutil.copy2(source_path, dest_path)
                else:
                    print(f"Warning: Could not find {item} in original profile.")

            # Write the creation date to the profile info file
            with open(profile_info_file, 'w') as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            print("Profile copy created successfully!")

        except Exception as e:
            print(f"Error copying profile: {e}")
            print("You may need to run this script with administrator privileges.")
            print("Try manually copying your Chrome profile to the 'chrome_profile_copy' directory.")

    def start_browser(self):
        """Start the browser if not already running"""
        if self.driver is None:
            service = Service()
            self.driver = webdriver.Chrome(service=service, options=self.chrome_options)

    def close_browser(self):
        """Close the browser if it's running"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def clean_url(self, url):
        """Clean URLs by removing query parameters and fragments"""
        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)

        # Extract the post ID from the path
        match = re.search(r'/posts/(\d+)', parsed_url.path)
        if match:
            post_id = match.group(1)
            # Extract group ID from path
            group_id_match = re.search(r'/groups/([^/]+)', parsed_url.path)
            group_id = group_id_match.group(1) if group_id_match else ""
            # Return a standardized URL format
            return f"https://www.facebook.com/groups/{group_id}/posts/{post_id}/"
        return url

    def extract_links(self, driver):
        """Extract post links from the current page"""
        previous_count = len(self.unique_posts)

        try:
            # Find all links with fresh DOM elements
            links = driver.find_elements(By.XPATH, "//a[contains(@href, '/posts/')]")
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if href and "/posts/" in href and "comment" not in href.lower() and "reply" not in href.lower():
                        clean_link = self.clean_url(href)
                        if clean_link:
                            self.unique_posts.add(clean_link)
                except Exception as e:
                    # Skip this link if it causes an error
                    continue
        except Exception as e:
            print(f"Error extracting links: {e}")

        # Calculate how many new posts we found in this scroll
        new_posts = len(self.unique_posts) - previous_count
        return new_posts

    def crawl_group(self, group_id, target_posts=300):
        """Crawl a Facebook group to extract post links"""
        self.start_browser()

        # Open the group page
        group_url = f'https://www.facebook.com/groups/{group_id}'
        self.driver.get(group_url)

        # Wait for the page to load completely
        WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Initial wait to let dynamic content load
        time.sleep(5)

        # Reset unique posts set
        self.unique_posts = set()
        scroll_count = 0
        consecutive_zero_posts = 0
        max_consecutive_zero = 5  # Max consecutive scrolls with zero new posts before giving up

        print(f"Starting to scroll until we find {target_posts} unique posts...")

        # Continue scrolling until we reach the target number of posts or hit a stopping condition
        while len(self.unique_posts) < target_posts and consecutive_zero_posts < max_consecutive_zero:
            scroll_count += 1
            print(f"Scroll #{scroll_count}")

            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Random wait between scrolls with more variation
            wait_time = random.uniform(2.0, 6.0)
            print(f"  Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            # Extract links and get the number of new posts found
            new_posts = self.extract_links(self.driver)

            # Update consecutive zero posts counter
            if new_posts == 0:
                consecutive_zero_posts += 1
                print(
                    f"  No new posts found. {consecutive_zero_posts}/{max_consecutive_zero} consecutive scrolls without new posts.")
            else:
                consecutive_zero_posts = 0  # Reset the counter if we found new posts

            print(
                f"  Found {new_posts} new posts in this scroll. Total unique posts: {len(self.unique_posts)}/{target_posts}")

            # Add a random short pause occasionally to seem more human-like
            if random.random() < 0.2:  # 20% chance
                extra_wait = random.uniform(1.0, 3.0)
                print(f"  Adding a short pause of {extra_wait:.2f} seconds...")
                time.sleep(extra_wait)

        # Get the page's HTML and save for backup
        html = self.driver.page_source
        with open("../../facebook_group_page.html", "w", encoding="utf-8") as f:
            f.write(html)

        # Additional regex extraction from saved HTML as a backup method
        patterns = [
            r'href="(https?://(?:www\.)?facebook\.com/groups/\d+/posts/\d+[^"]*)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                if not any(x in match.lower() for x in ['comment', 'reply', 'photo.php', 'reaction']):
                    clean_link = self.clean_url(match)
                    if clean_link:
                        self.unique_posts.add(clean_link)

        # Save the filtered links to links.txt
        output_file = os.path.join(os.getcwd(), "links.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for link in sorted(self.unique_posts):
                f.write(f"{link}\n")

        # Print final summary
        print("\nFinal Results:")
        print(f"- Total unique posts found: {len(self.unique_posts)}/{target_posts}")
        print(f"- Total scrolls performed: {scroll_count}")
        print(f"- Links saved to: {output_file}")

        if len(self.unique_posts) < target_posts:
            print(
                f"Note: Could not find {target_posts} unique posts. Stopped after {max_consecutive_zero} consecutive scrolls with no new posts.")

        # Return the set of unique posts found
        return self.unique_posts

    def extract_post_content(self, post_url):
        """Extract detailed content from a Facebook post"""
        self.start_browser()

        print(f"Extracting content from: {post_url}")

        # Navigate to the post
        try:
            self.driver.get(post_url)

            # Wait for the post content to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Add some random delay to seem more human-like
            time.sleep(random.uniform(2.0, 4.0))

            # Initialize data dictionary
            post_data = {
                "url": post_url,
                "date": None,
                "text_content": None,
                "images": [],
                "likes": None,
                "comments": None,
                "shares": None,
                "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Extract post date
            try:
                # Try to find the timestamp element
                date_elements = self.driver.find_elements(By.XPATH,
                                                          "//span[contains(@class, 'x1i10hfl') and contains(@class, 'xjbqb8w')]//a//span")
                if date_elements:
                    for element in date_elements:
                        if element.text and any(time_indicator in element.text.lower() for time_indicator in
                                                ["hr", "min", "h", "d", "day", "week", "month", "year", "yr", "sec"]):
                            post_data["date"] = element.text
                            break
            except Exception as e:
                print(f"Error extracting date: {e}")

            # Extract post text content
            try:
                # Look for the main post content div
                content_elements = self.driver.find_elements(By.XPATH,
                                                             "//div[contains(@class, 'xdj266r') and contains(@class, 'x11i5rnm')]")
                if content_elements:
                    post_data["text_content"] = content_elements[0].text
            except Exception as e:
                print(f"Error extracting text content: {e}")

            # Extract images
            try:
                # Look for image elements
                image_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'x1ey2m1c')]//img")
                for img in image_elements:
                    src = img.get_attribute("src")
                    if src and (src.startswith("http") and ("scontent" in src or "fbcdn" in src)):
                        post_data["images"].append(src)
            except Exception as e:
                print(f"Error extracting images: {e}")

            # Extract engagement metrics (likes, comments, shares)
            try:
                # For likes/reactions
                reaction_elements = self.driver.find_elements(By.XPATH,
                                                              "//span[contains(@class, 'xt0b8zv') and contains(@class, 'x2bj2ny')]")
                if reaction_elements:
                    text = reaction_elements[0].text
                    likes_match = re.search(r'(\d+(?:,\d+)*)', text)
                    if likes_match:
                        post_data["likes"] = likes_match.group(1)

                # For comments and shares
                engagement_elements = self.driver.find_elements(By.XPATH,
                                                                "//span[contains(@class, 'x193iq5w') and contains(@class, 'xeuugli')]")
                for element in engagement_elements:
                    text = element.text.lower()
                    if "comment" in text:
                        comments_match = re.search(r'(\d+(?:,\d+)*)\s*comment', text)
                        if comments_match:
                            post_data["comments"] = comments_match.group(1)
                    elif "share" in text:
                        shares_match = re.search(r'(\d+(?:,\d+)*)\s*share', text)
                        if shares_match:
                            post_data["shares"] = shares_match.group(1)
            except Exception as e:
                print(f"Error extracting engagement metrics: {e}")

            return post_data

        except Exception as e:
            print(f"Error processing post {post_url}: {e}")
            return {"url": post_url, "error": str(e)}

    def process_post_links(self, links=None, links_file=None, output_file="extracted_posts.json"):
        """Process a list of post links or links from a file"""
        if links is None:
            links = []

        # If links_file is provided, read links from it
        if links_file and os.path.exists(links_file):
            with open(links_file, "r", encoding="utf-8") as f:
                links.extend([line.strip() for line in f if line.strip()])

        if not links:
            print("No links to process.")
            return []

        print(f"Processing {len(links)} post links...")

        # Create and initialize browser if not already created
        self.start_browser()

        # Extract content from each post
        posts_data = []
        for i, link in enumerate(links):
            print(f"Processing post {i + 1}/{len(links)}: {link}")
            post_data = self.extract_post_content(link)
            posts_data.append(post_data)

            # Save progress after each post
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(posts_data, f, indent=4, ensure_ascii=False)

            # Random pause between posts to avoid detection
            if i < len(links) - 1:  # Don't pause after the last post
                pause_time = random.uniform(1.5, 5.0)
                print(f"Pausing for {pause_time:.2f} seconds...")
                time.sleep(pause_time)

        print(f"\nFinished processing all posts. Data saved to {output_file}")
        return posts_data

    def test_on_single_link(self, post_url):
        """Test the extraction on a single post link"""
        print(f"Testing extraction on: {post_url}")

        # Extract post content
        post_data = self.extract_post_content(post_url)

        # Print the results in a readable format
        print("\nExtraction Results:")
        print(f"URL: {post_data['url']}")
        print(f"Date: {post_data['date']}")
        print(f"Text Content: {post_data['text_content'][:100]}..." if post_data['text_content'] and len(
            post_data['text_content']) > 100 else f"Text Content: {post_data['text_content']}")
        print(f"Images: {len(post_data['images'])} found")
        for i, img in enumerate(post_data['images'][:3]):  # Show first 3 images
            print(f"  Image {i + 1}: {img[:60]}...")
        if len(post_data['images']) > 3:
            print(f"  ... and {len(post_data['images']) - 3} more images")
        print(f"Likes: {post_data['likes']}")
        print(f"Comments: {post_data['comments']}")
        print(f"Shares: {post_data['shares']}")
        print(f"Extraction Time: {post_data['extraction_time']}")

        # Save the test result to a JSON file
        test_output_file = "../../test_extraction_result.json"
        with open(test_output_file, "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=4, ensure_ascii=False)
        print(f"\nTest results saved to {test_output_file}")

        return post_data


def main():
    # Example usage
    user_data_dir = "C:\\Users\\IdoRavid\\AppData\\Local\\Google\\Chrome\\User Data"  # Change to your path

    # Ask if user wants to use profile copy or create a new one
    if os.path.exists(os.path.join(os.getcwd(), "chrome_profile_copy")):
        choice = input("Use existing Chrome profile copy or create a new one? (use/new): ")
        use_profile_copy = True
        force_new = choice.lower() == 'new'
    else:
        print("No Chrome profile copy found. Will create one.")
        use_profile_copy = True
        force_new = True

    # Create extractor instance with profile copy option
    extractor = FacebookPostExtractor(chrome_user_data_dir=user_data_dir, use_profile_copy=use_profile_copy)

    # If forcing a new profile copy, manually set it up
    if use_profile_copy and force_new:
        extractor._setup_profile_copy(force_new=True)

    # Choose one of the following operations:

    # 1. Crawl a group to collect posts
    # group_id = '462353464298357'  # Replace with your target group ID
    # post_links = extractor.crawl_group(group_id, target_posts=50)

    # 2. Test on a single link
    test_link = "https://www.facebook.com/groups/325992450444/posts/10170764157255445/"  # Replace with a real post link
    extractor.test_on_single_link(test_link)

    # 3. Process links from a file
    # extractor.process_post_links(links_file="links.txt", output_file="extracted_posts.json")

    # Close browser when done
    extractor.close_browser()


if __name__ == "__main__":
    main()