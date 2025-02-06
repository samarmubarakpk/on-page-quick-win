import streamlit as st
import pandas as pd
import requests
from typing import List, Dict
import re
import string
from selectolax.parser import HTMLParser
import html5lib

from app import clean_to_english, convert_to_numeric, format_avg_position, format_ctr, get_top_queries_per_url

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
            'h2': self.h2s[:5],  # Limit to first 5 H2s
            'content': ' '.join(self.paragraphs)
        }

def scrape_content(url: str, content_wrapper_class: str = None) -> Dict:
    """Scrape content from a URL using selectolax"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        
        tree = HTMLParser(response.text)
        results = {
            'h1': [],
            'h2': [],
            'content': '',
            'title': '',
            'meta_description': ''
        }
        
        # Extract title and meta description
        title_elem = tree.css_first('title')
        results['title'] = title_elem.text().strip() if title_elem else ''
        
        meta_desc_elem = tree.css_first('meta[name="description"]')
        if meta_desc_elem:
            results['meta_description'] = meta_desc_elem.attributes.get('content', '').strip()
        
        if content_wrapper_class:
            content_area = tree.css_first(f'div.{content_wrapper_class}')
            root = content_area if content_area else tree
        else:
            root = tree
            
        for selector in ['script', 'style', 'nav', 'header', 'footer']:
            for elem in root.css(selector):
                elem.remove()
        
        results['h1'] = [node.text().strip() for node in root.css('h1') if node.text().strip()]
        h2s = [node.text().strip() for node in root.css('h2') if node.text().strip()]
        results['h2'] = h2s[:5]  # Limit to first 5 H2s
        
        content_texts = [node.text().strip() for node in root.css('p, li') if node.text().strip()]
        results['content'] = ' '.join(content_texts)
        
        return results
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return {'h1': [], 'h2': [], 'content': '', 'title': '', 'meta_description': ''}

def analyze_url(url: str, query: str, content_wrapper: str = None) -> Dict:
    """Analyze a URL for SEO elements and keyword usage"""
    try:
        default_response = {
            'url': url,
            'title': '',
            'title_contains': False,
            'meta_description': '',
            'meta_contains': False,
            'h1_matches': [],
            'h2_checks': [],
            'content_matches': [],
            'has_content': False,
            'error': None
        }
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code != 200:
            default_response['error'] = f"{response.status_code} {response.reason}"
            return default_response
            
        content = scrape_content(url, content_wrapper)
        
        if not content or not any([content['h1'], content['h2'], content['content']]):
            default_response['error'] = "No content found"
            return default_response
        
        clean_query = clean_to_english(query.lower())
        query_words = set(clean_query.split())
        
        # Check title and meta description
        title_text = content.get('title', '')
        clean_title = clean_to_english(title_text.lower())
        title_contains = any(word in clean_title for word in query_words)
        
        meta_desc_text = content.get('meta_description', '')
        clean_meta = clean_to_english(meta_desc_text.lower())
        meta_contains = any(word in clean_meta for word in query_words)
        
        # Check H1 matches
        h1_matches = []
        for h1 in content['h1']:
            clean_h1 = clean_to_english(h1.lower())
            if any(word in clean_h1 for word in query_words):
                h1_matches.append(h1)
                
        # Check H2s (up to 5)
        h2_checks = []
        for h2 in content.get('h2', [])[:5]:
            clean_h2 = clean_to_english(h2.lower())
            contains = any(word in clean_h2 for word in query_words)
            h2_checks.append((h2, contains))
        
        # Content matches
        content_matches = []
        if content['content']:
            clean_content = clean_to_english(content['content'].lower())
            sentences = [s.strip() for s in re.split(r'[.!?]+', clean_content)]
            content_matches = [sent for sent in sentences if sent and any(word in sent.lower() for word in query_words)]
        
        return {
            'url': url,
            'title': title_text,
            'title_contains': title_contains,
            'meta_description': meta_desc_text,
            'meta_contains': meta_contains,
            'h1_matches': h1_matches,
            'h2_checks': h2_checks,
            'content_matches': content_matches[:5],
            'has_content': bool(content['content']),
            'error': None
        }
        
    except requests.exceptions.RequestException as e:
        default_response['error'] = f"Request error: {str(e)}"
        return default_response
    except Exception as e:
        default_response['error'] = f"Error: {str(e)}"
        return default_response

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
                
                # Exclude rows containing any branded terms in the Query column
                if branded_terms:
                    pattern = '|'.join([re.escape(term) for term in branded_terms])
                    df = df[~df['Query'].str.contains(pattern, case=False, na=False)]

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
                        h2_checks = analysis.get('h2_checks', [])
                        result_entry = {
                            'URL': row['Landing Page'],
                            'Query': row['Query'],
                            'Clicks': int(row['Clicks']),
                            'Impressions': int(row['Impressions']),
                            'Avg. Position': row['Avg. Pos'],
                            'CTR': format_ctr(row['Clicks'] / row['Impressions'] * 100 if row['Impressions'] > 0 else 0),
                            'Title': analysis.get('title', ''),
                            'Title Contains': analysis.get('title_contains', False),
                            'Meta Description': analysis.get('meta_description', ''),
                            'Meta Contains': analysis.get('meta_contains', False),
                            'H1': analysis.get('h1_matches', []),
                            'H1 Contains': bool(analysis.get('h1_matches', [])),
                            'H2-1': h2_checks[0][0] if len(h2_checks) > 0 else '',
                            'H2-1 Contains': h2_checks[0][1] if len(h2_checks) > 0 else False,
                            'H2-2': h2_checks[1][0] if len(h2_checks) > 1 else '',
                            'H2-2 Contains': h2_checks[1][1] if len(h2_checks) > 1 else False,
                            'H2-3': h2_checks[2][0] if len(h2_checks) > 2 else '',
                            'H2-3 Contains': h2_checks[2][1] if len(h2_checks) > 2 else False,
                            'H2-4': h2_checks[3][0] if len(h2_checks) > 3 else '',
                            'H2-4 Contains': h2_checks[3][1] if len(h2_checks) > 3 else False,
                            'H2-5': h2_checks[4][0] if len(h2_checks) > 4 else '',
                            'H2-5 Contains': h2_checks[4][1] if len(h2_checks) > 4 else False,
                            'Copy': analysis.get('content_matches', []),
                            'Copy Contains': bool(analysis.get('content_matches', []))
                        }
                        results.append(result_entry)

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
