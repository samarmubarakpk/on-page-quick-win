import streamlit as st
import pandas as pd
import requests
from typing import List, Dict
import re
import string
from selectolax.parser import HTMLParser

def extract_text(element):
    """Extract text from an element and its children"""
    return ''.join(element.itertext()).strip()

class ContentParser:
    def __init__(self, content_wrapper=None):
        self.h1s = []
        self.h2s = []
        self.paragraphs = []
        self.content_wrapper = content_wrapper
        self.in_wrapper = False if content_wrapper else True
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer'}
        self.in_skip = False
        self.current_content = []
        
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        
        # Skip unwanted sections
        if tag in self.skip_tags:
            self.in_skip = True
            return
            
        if self.in_skip:
            return
            
        # Handle content wrapper
        if self.content_wrapper and tag == 'div' and 'class' in attrs:
            classes = attrs['class'].split()
            if self.content_wrapper in classes:
                self.in_wrapper = True
                
        self.current_tag = tag
        self.current_content = []
        
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip = False
            return
            
        if self.in_skip:
            return
            
        if not self.in_wrapper:
            return
            
        content = ''.join(self.current_content).strip()
        if content:
            if tag == 'h1':
                self.h1s.append(content)
            elif tag == 'h2':
                self.h2s.append(content)
            elif tag in ('p', 'li'):
                self.paragraphs.append(content)
                
        self.current_tag = None
        self.current_content = []
        
    def handle_data(self, data):
        if self.in_skip:
            return
            
        if not self.in_wrapper:
            return
            
        if self.current_tag in ('h1', 'h2', 'p', 'li'):
            self.current_content.append(data.strip())
            
    def get_content(self):
        return {
            'h1': self.h1s,
            'h2': self.h2s,
            'content': ' '.join(self.paragraphs)
        }

def scrape_content(url: str, content_wrapper_class: str = None) -> Dict:
    """Scrape content from a URL using selectolax"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with selectolax
        tree = HTMLParser(response.text)
        
        # Initialize results
        results = {
            'h1': [],
            'h2': [],
            'content': []
        }
        
        # Find content area based on wrapper class if specified
        if content_wrapper_class:
            content_area = tree.css_first(f'div.{content_wrapper_class}')
            root = content_area if content_area else tree
        else:
            root = tree
            
        # Remove unwanted elements
        for selector in ['script', 'style', 'nav', 'header', 'footer']:
            for elem in root.css(selector):
                elem.remove()
        
        # Extract content
        results['h1'] = [
            node.text().strip()
            for node in root.css('h1')
            if node.text().strip()
        ]
        
        results['h2'] = [
            node.text().strip()
            for node in root.css('h2')
            if node.text().strip()
        ]
        
        # Get paragraphs and list items
        content_texts = [
            node.text().strip()
            for node in root.css('p, li')
            if node.text().strip()
        ]
        
        # Join content
        results['content'] = ' '.join(content_texts)
        
        return results
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return {'h1': [], 'h2': [], 'content': ''}

def scrape_content_html5lib(url: str, content_wrapper_class: str = None) -> Dict:
    """Scrape content from a URL using html5lib"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with html5lib
        document = html5lib.parse(response.text, namespaceHTMLElements=False)
        
        # Initialize results
        results = {
            'h1': [],
            'h2': [],
            'content': []
        }
        
        # Find content wrapper if specified
        if content_wrapper_class:
            wrapper = None
            for elem in document.iter('div'):
                if elem.get('class') and content_wrapper_class in elem.get('class').split():
                    wrapper = elem
                    break
            root = wrapper if wrapper else document
        else:
            root = document
            
        # Remove unwanted elements
        for elem in root.findall('.//script'):
            elem.clear()
        for elem in root.findall('.//style'):
            elem.clear()
        for elem in root.findall('.//nav'):
            elem.clear()
        for elem in root.findall('.//header'):
            elem.clear()
        for elem in root.findall('.//footer'):
            elem.clear()
            
        # Extract content
        for elem in root.iter():
            if elem.tag == 'h1':
                text = extract_text(elem)
                if text:
                    results['h1'].append(text)
            elif elem.tag == 'h2':
                text = extract_text(elem)
                if text:
                    results['h2'].append(text)
            elif elem.tag in ('p', 'li'):
                text = extract_text(elem)
                if text:
                    results['content'].append(text)
        
        # Join content
        results['content'] = ' '.join(results['content'])
        
        return results
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return {'h1': [], 'h2': [], 'content': ''}

def analyze_url(url: str, query: str, content_wrapper: str = None) -> Dict:
    """Analyze a URL for SEO elements and keyword usage"""
    try:
        content = scrape_content(url, content_wrapper)
        
        # Clean and prepare query
        clean_query = clean_to_english(query.lower())
        query_words = set(clean_query.split())
        
        # Check content matches
        h1_matches = []
        for h1 in content['h1']:
            clean_h1 = clean_to_english(h1.lower())
            if any(word in clean_h1 for word in query_words):
                h1_matches.append(h1)
                
        h2_matches = []
        for h2 in content['h2']:
            clean_h2 = clean_to_english(h2.lower())
            if any(word in clean_h2 for word in query_words):
                h2_matches.append(h2)
                
        # Find relevant content snippets
        content_matches = []
        if content['content']:
            clean_content = clean_to_english(content['content'].lower())
            sentences = [s.strip() for s in re.split(r'[.!?]+', clean_content)]
            content_matches = [
                sent for sent in sentences
                if sent and any(word in sent.lower() for word in query_words)
            ]
        
        return {
            'h1_matches': h1_matches,
            'h2_matches': h2_matches,
            'content_matches': content_matches[:5],  # Show top 5 matches
            'has_content': bool(content['content'])
        }
        
    except Exception as e:
        st.error(f"Error analyzing {url}: {str(e)}")
        return {
            'h1_matches': [],
            'h2_matches': [],
            'content_matches': [],
            'has_content': False
        }

def clean_to_english(text: str) -> str:
    """Clean text to basic English characters"""
    # Remove punctuation except periods for sentence splitting
    translator = str.maketrans('', '', string.punctuation.replace('.', ''))
    return text.translate(translator)

def is_branded_query(query: str, branded_terms: List[str]) -> bool:
    """Check if a query contains branded terms"""
    if not branded_terms:
        return False
        
    query_clean = clean_to_english(query.lower())
    return any(
        clean_to_english(term.lower()) in query_clean
        for term in branded_terms
    )

def format_average_position(pos: str) -> float:
    """Format average position to decimal"""
    try:
        if isinstance(pos, (int, float)):
            return float(pos)
        if isinstance(pos, str):
            # Remove any non-numeric characters except decimal point
            clean_pos = ''.join(c for c in pos if c.isdigit() or c == '.')
            return float(clean_pos) if clean_pos else 0.0
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def convert_to_numeric(series):
    """Safely convert a series to numeric, handling both string and numeric inputs"""
    if series.dtype.kind in 'iuf':  # If already integer, unsigned int, or float
        return series
    return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def get_top_queries_per_url(df: pd.DataFrame, max_queries: int = 10) -> pd.DataFrame:
    """Get top queries by clicks for each unique URL"""
    # Convert numeric columns safely
    df['Clicks'] = convert_to_numeric(df['Clicks'])
    df['Impressions'] = convert_to_numeric(df['Impressions'])
    
    # Remove rows with 0 clicks or invalid numbers
    df = df[df['Clicks'].notna() & (df['Clicks'] > 0)].copy()
    
    # Remove branded queries if any were marked
    df['is_branded'] = df['Query'].apply(lambda x: 'BRANDED' in str(x).upper())
    df = df[~df['is_branded']]
    
    # Group by Landing Page and get top queries
    results = []
    for url in df['Landing Page'].unique():
        url_data = df[df['Landing Page'] == url]
        
        # Sort by Clicks (descending) and take top N queries
        top_queries = url_data.nlargest(max_queries, 'Clicks')
        
        if not top_queries.empty:
            results.append(top_queries)
    
    if not results:
        return pd.DataFrame()  # Return empty DataFrame if no results
        
    return pd.concat(results)

def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL"""
    try:
        return bool(url and isinstance(url, str) and url.startswith('http'))
    except:
        return False

def clean_gsc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean GSC data by removing invalid rows and formatting numbers"""
    # Store original length
    original_len = len(df)
    
    # Filter rows with valid URLs
    df = df[df['Landing Page'].apply(is_valid_url)]
    
    # Convert clicks and impressions to numeric, replacing non-numeric with 0
    df['Clicks'] = convert_to_numeric(df['Clicks'])
    df['Impressions'] = convert_to_numeric(df['Impressions'])
    
    # Remove rows where Query is empty or non-string
    df = df[df['Query'].notna() & df['Query'].astype(str).str.strip().astype(bool)]
    
    # Remove rows with obviously invalid position values
    df = df[pd.to_numeric(df['Avg. Pos'].astype(str).str.replace(',', '').str.replace('.', ''), errors='coerce').notna()]
    
    # Report cleaning results
    rows_removed = original_len - len(df)
    if rows_removed > 0:
        st.warning(f"Removed {rows_removed} invalid rows from the dataset (non-URL content or invalid data)")
    
    return df

def format_ctr(value: float) -> str:
    """Format CTR as percentage with 2 decimal places"""
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return "0.00%"

def format_avg_position(position_str: str) -> float:
    """Format average position to have correct decimal places"""
    try:
        # Handle empty or invalid input
        if not position_str or str(position_str).strip() == '0':
            return 0.0
            
        # Convert to string and clean it
        position_str = str(position_str).replace(',', '').strip()
        
        # Try to convert to float directly
        try:
            return round(float(position_str), 1)
        except ValueError:
            # If that fails, try to extract the first number
            numbers = re.findall(r'\d+\.?\d*', position_str)
            if numbers:
                return round(float(numbers[0]), 1)
            
        return 0.0
            
    except (ValueError, TypeError, IndexError):
        return 0.0

def main():
    st.set_page_config(page_title="On Page Quick Wins", layout="wide")
    
    # Initialize session state for results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'not_found_urls' not in st.session_state:
        st.session_state.not_found_urls = None
    if 'other_errors' not in st.session_state:
        st.session_state.other_errors = None
    
    st.title("On Page Quick Wins")
    st.write("Upload your GSC performance report to analyze keyword usage in your pages.")
    
    # Add content wrapper input
    content_wrapper = st.text_input(
        "Content Wrapper Class (Optional)",
        help="Enter the HTML class name that wraps your main content (e.g., 'blog-content-wrapper' or 'product-description'). "
        "This helps extract only relevant content and ignore navigation, popups, etc."
    )
    
    # Add branded terms input
    st.subheader("Branded Terms")
    branded_terms_input = st.text_area(
        "Enter your branded terms (one per line) to exclude from analysis:",
        help="Enter terms related to your brand that you want to exclude from the analysis. For example:\nApple\niPhone\nMacBook"
    )
    branded_terms = [term.strip() for term in branded_terms_input.split('\n') if term.strip()]
    
    uploaded_file = st.file_uploader("Upload your GSC Performance Export (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # Try different CSV reading configurations
            try:
                # First try with semicolon separator
                df = pd.read_csv(uploaded_file, sep=';')
            except pd.errors.ParserError:
                # If that fails, try with comma separator
                uploaded_file.seek(0)  # Reset file pointer
                try:
                    df = pd.read_csv(uploaded_file)
                except pd.errors.ParserError:
                    # If that fails too, try with more flexible settings
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(
                        uploaded_file,
                        sep=None,  # Detect separator
                        engine='python',
                        on_bad_lines='skip',
                        encoding='utf-8',
                        skipinitialspace=True
                    )
            
            # Drop empty columns
            df = df.dropna(axis=1, how='all')
            
            # Check if we need to run the analysis
            run_analysis = st.button("Run Analysis")
            
            if run_analysis:
                st.session_state.analysis_results = None  # Clear previous results
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Ensure required columns exist
                required_columns = {'Query', 'Landing Page', 'Clicks', 'Impressions', 'Avg. Pos'}
                if not all(col in df.columns for col in required_columns):
                    st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                    st.write("Found columns:", ', '.join(df.columns))
                    return
                
                # Format Avg. Pos first before any other numeric conversions
                df['Avg. Pos'] = df['Avg. Pos'].apply(format_avg_position)
                
                # Convert other numeric columns safely
                df['Clicks'] = convert_to_numeric(df['Clicks'])
                df['Impressions'] = convert_to_numeric(df['Impressions'])
                
                # Drop rows with invalid numeric values
                df = df.dropna(subset=['Clicks', 'Impressions', 'Avg. Pos'])
                
                # Get top queries per URL
                unique_urls = len(df['Landing Page'].unique())
                top_queries = get_top_queries_per_url(df)
                
                if top_queries.empty:
                    st.warning("No valid queries found after filtering.")
                    return
                
                total_queries = len(top_queries)
                st.write(f"Analyzing top queries for {unique_urls} URLs (Total queries: {total_queries})")
                
                # Initialize collections for results and errors
                results = []
                not_found_urls = set()  # Using set to avoid duplicates
                other_errors = []
                
                # Create progress bar with proper chunking
                total_analyses = len(top_queries)
                chunk_size = max(1, total_analyses // 100)  # Update progress every 1%
                progress_bar = st.progress(0)
                
                for idx, row in top_queries.iterrows():
                    # Update progress every chunk_size iterations
                    if idx % chunk_size == 0:
                        progress = min(1.0, idx / total_analyses)
                        progress_bar.progress(progress)
                    
                    analysis = analyze_url(row['Landing Page'], row['Query'], content_wrapper)
                    
                    if not analysis['has_content']:
                        not_found_urls.add(analysis['url'])
                    else:
                        results.append({
                            'URL': row['Landing Page'],
                            'Query': row['Query'],
                            'Clicks': int(row['Clicks']),
                            'Impressions': int(row['Impressions']),
                            'Avg. Position': row['Avg. Pos'],
                            'CTR': format_ctr(row['Clicks'] / row['Impressions'] * 100 if row['Impressions'] > 0 else 0),
                            'Title': '',
                            'Title Contains': False,
                            'Meta Description': '',
                            'Meta Contains': False,
                            'H1': analysis['h1_matches'],
                            'H1 Contains': bool(analysis['h1_matches']),
                            'H2-1': analysis['h2_matches'],
                            'H2-1 Contains': bool(analysis['h2_matches']),
                            'H2-2': '',
                            'H2-2 Contains': False,
                            'H2-3': '',
                            'H2-3 Contains': False,
                            'H2-4': '',
                            'H2-4 Contains': False,
                            'H2-5': '',
                            'H2-5 Contains': False,
                            'Copy': analysis['content_matches'],
                            'Copy Contains': bool(analysis['content_matches'])
                        })
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.not_found_urls = not_found_urls
                st.session_state.other_errors = other_errors
            
            # Display results if they exist in session state
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                not_found_urls = st.session_state.not_found_urls
                other_errors = st.session_state.other_errors
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Group results by URL for better visualization
                    st.subheader("Analysis Results")
                    
                    # Add download button for all results at the top
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Analysis Report (All URLs)",
                        data=csv,
                        file_name="seo_analysis_report_all.csv",
                        mime="text/csv",
                        help="Download the complete analysis for all URLs in a single CSV file"
                    )
                    
                    # Show individual URL tables
                    for url in results_df['URL'].unique():
                        st.write(f"### {url}")
                        url_results = results_df[results_df['URL'] == url]
                        st.dataframe(url_results, use_container_width=True)
                        
                        # Individual URL download (optional)
                        url_csv = url_results.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {url.split('/')[-1] or 'homepage'} Analysis",
                            data=url_csv,
                            file_name=f"seo_analysis_{url.split('/')[-1] or 'homepage'}.csv",
                            mime="text/csv",
                            key=f"download_{url}"  # Unique key for each button
                        )
                    
                    # Show error summary if there were any issues
                    if not_found_urls or other_errors:
                        st.subheader("⚠️ Analysis Warnings")
                        
                        # Display 404 errors
                        if not_found_urls:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.warning(f"Found {len(not_found_urls)} URLs returning 404 Not Found")
                            with col2:
                                # Create CSV for 404 URLs
                                not_found_df = pd.DataFrame({'URL': list(not_found_urls)})
                                csv = not_found_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download 404 URLs",
                                    data=csv,
                                    file_name="404_not_found_urls.csv",
                                    mime="text/csv",
                                )
                        
                        # Display other errors if any
                        if other_errors:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.warning(f"Found {len(other_errors)} URLs with other errors")
                            with col2:
                                # Create CSV for other errors
                                other_errors_df = pd.DataFrame(other_errors)
                                csv = other_errors_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Error Details",
                                    data=csv,
                                    file_name="url_errors.csv",
                                    mime="text/csv",
                                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
