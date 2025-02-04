import streamlit as st
import pandas as pd
import requests
from html.parser import HTMLParser
import re
from typing import List, Dict
import html
import string

class SEOParser(HTMLParser):
    def __init__(self, content_wrapper=None):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.h1 = []
        self.h2 = []
        self.main_content = []
        self.current_tag = None
        self.current_text = []
        self.reading_content = False
        self.content_wrapper = content_wrapper
        self.in_wrapper = False if content_wrapper else True
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript'}
        self.current_skip = False
        self.wrapper_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        
        # Skip unwanted tags
        if tag in self.skip_tags:
            self.current_skip = True
            return

        if self.current_skip:
            return

        # Handle content wrapper
        if tag == 'div' and self.content_wrapper:
            if 'class' in attrs and self.content_wrapper in attrs['class']:
                self.in_wrapper = True
                self.wrapper_depth += 1
            elif self.in_wrapper:
                self.wrapper_depth += 1

        # Track current tag for data handling
        self.current_tag = tag
        self.current_text = []

        # Handle meta tags
        if tag == 'meta' and 'name' in attrs and attrs['name'] == 'description':
            self.meta_description = attrs.get('content', '')

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = False
            return

        if self.current_skip:
            return

        # Handle content wrapper
        if tag == 'div' and self.in_wrapper:
            self.wrapper_depth -= 1
            if self.wrapper_depth == 0:
                self.in_wrapper = False

        # Process collected text
        if self.current_text and self.in_wrapper:
            text = ''.join(self.current_text).strip()
            if text:
                if tag == 'h1':
                    self.h1.append(text)
                elif tag == 'h2':
                    self.h2.append(text)
                elif tag in ['p', 'li']:
                    self.main_content.append(text)

        self.current_tag = None
        self.current_text = []

    def handle_data(self, data):
        if self.current_skip:
            return

        data = data.strip()
        if not data:
            return

        if self.current_tag == 'title':
            self.title = data
        elif self.in_wrapper and self.current_tag:
            self.current_text.append(data)

    def handle_entityref(self, name):
        if self.current_skip:
            return
        
        if self.current_tag:
            self.current_text.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if self.current_skip:
            return
            
        if self.current_tag:
            self.current_text.append(html.unescape(f"&#{name};"))

    def get_data(self):
        return ' '.join(self.main_content)

def scrape_content(url: str, content_wrapper_class: str = None) -> dict:
    """Scrape content from a URL, optionally within a specific class wrapper"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        parser = SEOParser(content_wrapper=content_wrapper_class)
        parser.feed(response.text)
        
        # Get all H2s from the main content area, excluding navigation/footer
        h2s = parser.h2
        
        # Extract text from all H2s
        h2_texts = h2s
        
        # Get main content
        content = parser.get_data()
            
        return {
            'h1': [],  # We'll handle H1s separately
            'h2': h2_texts,  # Return all H2s found
            'content': content,
            'url': url
        }
        
    except Exception as e:
        return {'error': str(e), 'url': url}

def analyze_url(url: str, query: str, content_wrapper: str = None) -> Dict:
    """Analyze a URL for SEO elements and keyword usage"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 404:
            return {'error': '404', 'url': url}
        response.raise_for_status()
        
        parser = SEOParser(content_wrapper)
        parser.feed(response.text)
        
        # Get the first 5 H2s or empty strings if not enough
        h2s = parser.h2[:5] + [''] * (5 - len(parser.h2))
        
        # Join main content with spaces, properly decode HTML entities, clean it, and filter to English
        main_content = parser.get_data()
        main_content = html.unescape(main_content)  # Properly decode HTML entities
        main_content = clean_to_english(main_content)  # Filter to English characters
        
        # Clean all text fields to English characters
        title = clean_to_english(parser.title)
        meta_description = clean_to_english(parser.meta_description)
        h1 = clean_to_english(parser.h1[0] if parser.h1 else '')
        h2s = [clean_to_english(h) for h in h2s]
        query = clean_to_english(query)  # Clean the query too for consistent matching
        
        # Function to check if query appears in text (case insensitive)
        def contains_query(text):
            return query.lower() in clean_to_english(text).lower() if text else False
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'title_contains': contains_query(title),
            'meta_description': meta_description,
            'meta_contains': contains_query(meta_description),
            'h1': h1,
            'h1_contains': contains_query(h1),
            'h2_1': h2s[0],
            'h2_1_contains': contains_query(h2s[0]),
            'h2_2': h2s[1],
            'h2_2_contains': contains_query(h2s[1]),
            'h2_3': h2s[2],
            'h2_3_contains': contains_query(h2s[2]),
            'h2_4': h2s[3],
            'h2_4_contains': contains_query(h2s[3]),
            'h2_5': h2s[4],
            'h2_5_contains': contains_query(h2s[4]),
            'main_content': main_content[:500] + '...' if len(main_content) > 500 else main_content,
            'content_contains': contains_query(main_content)
        }
    except requests.exceptions.RequestException as e:
        return {'error': str(e), 'url': url}
    except Exception as e:
        return {'error': str(e), 'url': url}

def clean_to_english(text: str) -> str:
    """Remove non-English characters and clean the text"""
    if not text:
        return ""
    
    # Define valid characters (English letters, numbers, and basic punctuation)
    valid_chars = string.ascii_letters + string.digits + string.punctuation + ' '
    
    # Replace common Unicode quotes and dashes with ASCII equivalents
    replacements = {
        '"': '"',  # Smart quotes
        '"': '"',
        ''': "'",  # Smart apostrophes
        ''': "'",
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '…': '...' # Ellipsis
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Keep only valid characters
    text = ''.join(c for c in text if c in valid_chars)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text

def convert_to_numeric(series):
    """Safely convert a series to numeric, handling both string and numeric inputs"""
    if series.dtype.kind in 'iuf':  # If already integer, unsigned int, or float
        return series
    return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def is_branded_query(query: str, branded_terms: List[str]) -> bool:
    """Check if a query contains any branded terms"""
    query = query.lower()
    return any(brand.lower() in query for brand in branded_terms)

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
            
        # Convert to string and remove any commas
        position_str = str(position_str).replace(',', '').strip()
        
        # If it's a decimal number (either with . or ,)
        if '.' in position_str:
            return round(float(position_str), 1)
            
        # If it's a whole number
        if position_str.isdigit():
            return float(position_str)
            
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
                    
                    if analysis.get('error') == '404':
                        not_found_urls.add(analysis['url'])
                    elif 'error' in analysis:
                        other_errors.append({
                            'url': analysis['url'],
                            'error': analysis['error']
                        })
                    else:
                        results.append({
                            'URL': row['Landing Page'],
                            'Query': row['Query'],
                            'Clicks': int(row['Clicks']),
                            'Impressions': int(row['Impressions']),
                            'Avg. Position': row['Avg. Pos'],
                            'CTR': format_ctr(row['Clicks'] / row['Impressions'] * 100 if row['Impressions'] > 0 else 0),
                            'Title': analysis['title'],
                            'Title Contains': analysis['title_contains'],
                            'Meta Description': analysis['meta_description'],
                            'Meta Contains': analysis['meta_contains'],
                            'H1': analysis['h1'],
                            'H1 Contains': analysis['h1_contains'],
                            'H2-1': analysis['h2_1'],
                            'H2-1 Contains': analysis['h2_1_contains'],
                            'H2-2': analysis['h2_2'],
                            'H2-2 Contains': analysis['h2_2_contains'],
                            'H2-3': analysis['h2_3'],
                            'H2-3 Contains': analysis['h2_3_contains'],
                            'H2-4': analysis['h2_4'],
                            'H2-4 Contains': analysis['h2_4_contains'],
                            'H2-5': analysis['h2_5'],
                            'H2-5 Contains': analysis['h2_5_contains'],
                            'Copy': analysis['main_content'],
                            'Copy Contains': analysis['content_contains']
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
