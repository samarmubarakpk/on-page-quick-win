import streamlit as st
import pandas as pd
import requests
from html.parser import HTMLParser
import re
from typing import List, Dict

class SEOParser(HTMLParser):
    def __init__(self, content_wrapper=None):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.h1 = []
        self.h2 = []
        self.main_content = []
        self.current_tag = None
        self.reading_content = False
        self.content_wrapper = content_wrapper
        self.in_wrapper = False if content_wrapper else True  # If no wrapper specified, always collect content
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript'}
        self.current_skip = False
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        # Check if we're entering the content wrapper
        if self.content_wrapper and tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class'].split()
            if self.content_wrapper in classes:
                self.in_wrapper = True
        
        # Skip unwanted elements
        if tag in self.skip_tags:
            self.current_skip = True
            return
        
        if tag == 'title':
            self.reading_content = True
        elif tag == 'meta' and attrs_dict.get('name', '').lower() == 'description':
            self.meta_description = attrs_dict.get('content', '')
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = False
        
        # Check if we're exiting the content wrapper
        if self.content_wrapper and tag == 'div' and self.in_wrapper:
            self.in_wrapper = False
        
        if tag == 'title':
            self.reading_content = False
        self.current_tag = None
    
    def handle_data(self, data):
        if self.current_skip:
            return
            
        data = data.strip()
        if not data:
            return
        
        if self.reading_content and self.current_tag == 'title':
            self.title = data
        elif self.current_tag == 'h1':
            self.h1.append(data)
        elif self.current_tag == 'h2':
            self.h2.append(data)
        # Only collect main content if we're in the wrapper (or if no wrapper specified)
        elif self.in_wrapper and self.current_tag not in {'title', 'h1', 'h2'}:
            self.main_content.append(data)

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
        
        # Join main content with spaces and clean it
        main_content = ' '.join(parser.main_content).strip()
        
        # Function to check if query appears in text (case insensitive)
        def contains_query(text):
            return query.lower() in text.lower() if text else False
        
        return {
            'success': True,
            'url': url,
            'title': parser.title,
            'title_contains': contains_query(parser.title),
            'meta_description': parser.meta_description,
            'meta_contains': contains_query(parser.meta_description),
            'h1': parser.h1[0] if parser.h1 else '',
            'h1_contains': contains_query(parser.h1[0] if parser.h1 else ''),
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

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def check_keyword_presence(text: str, keyword: str) -> bool:
    """Check if keyword is present in text (case insensitive)"""
    if not text or not keyword:
        return False
    return keyword.lower() in text.lower()

def get_top_queries_per_url(df: pd.DataFrame, max_queries: int = 10) -> pd.DataFrame:
    """Get top queries by clicks for each unique URL"""
    # Remove rows with 0 clicks
    df = df[df['Clicks'] > 0].copy()
    
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

def is_branded_query(query: str, branded_terms: List[str]) -> bool:
    """Check if a query contains any branded terms"""
    query = query.lower()
    return any(brand.lower() in query for brand in branded_terms if brand.strip())

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
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce').fillna(0).astype(int)
    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').fillna(0).astype(int)
    
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
        # Remove any commas or dots from the string
        clean_str = str(position_str).replace(',', '').replace('.', '')
        
        # Get the first digits before any potential decimals
        first_digits = clean_str[:2] if len(clean_str) > 1 else clean_str
        
        # If it starts with a single digit (1-9)
        if len(first_digits) == 1 or (len(first_digits) == 2 and first_digits[0] == '0'):
            # Take first digit and next digit as decimal
            return round(float(f"{first_digits[0]}.{clean_str[1]}"), 1)
        else:
            # For numbers starting with 2 or more digits
            # Take first two digits and third digit as decimal
            return round(float(f"{first_digits}.{clean_str[2]}"), 1)
            
    except (ValueError, TypeError, IndexError):
        return 0.0

def main():
    st.set_page_config(page_title="On Page Quick Wins", layout="wide")
    
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
    
    if branded_terms:
        st.info(f"The following branded terms will be excluded: {', '.join(branded_terms)}")
    
    uploaded_file = st.file_uploader("Upload GSC Performance Report (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # Try different CSV reading configurations
            try:
                # First try with semicolon separator
                df = pd.read_csv(uploaded_file, sep=';')
            except pd.errors.ParserError:
                # If that fails, try with different settings
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(
                    uploaded_file,
                    sep=';',
                    encoding='utf-8',
                    on_bad_lines='skip',
                    skipinitialspace=True,
                    engine='python'
                )
            
            # Drop empty columns (columns with all NaN values)
            df = df.dropna(axis=1, how='all')
            
            # Check for required columns with exact names
            required_columns = ['Query', 'Landing Page', 'Clicks', 'Impressions', 'Avg. Pos']
            
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                st.write("Found columns:", ', '.join([col for col in df.columns if not pd.isna(col)]))
                return
            
            # Clean the data
            df = clean_gsc_data(df)
            
            if len(df) == 0:
                st.error("No valid data rows found after cleaning. Please check your CSV file.")
                return
            
            # Format numeric columns
            df['Clicks'] = df['Clicks'].astype(int)
            df['Impressions'] = df['Impressions'].astype(int)
            df['Avg. Pos'] = df['Avg. Pos'].apply(format_avg_position)
            if 'URL CTR' in df.columns:
                df['URL CTR'] = df['URL CTR'].apply(lambda x: float(str(x).replace('%', '')) if pd.notna(x) else 0)
                df['URL CTR'] = df['URL CTR'].apply(format_ctr)
            
            # Filter out branded queries first
            if branded_terms:
                original_count = len(df)
                df['is_branded'] = df['Query'].apply(lambda x: is_branded_query(x, branded_terms))
                df = df[~df['is_branded']]
                filtered_count = len(df)
                st.write(f"Filtered out {original_count - filtered_count} branded queries.")
            
            # Show data preview of cleaned and filtered data
            st.subheader("Data Preview")
            st.write("First few rows of your cleaned and filtered data:")
            preview_df = df.head()
            # Format the preview dataframe
            preview_df = preview_df.drop('is_branded', axis=1, errors='ignore')
            st.dataframe(preview_df, use_container_width=True)
            
            # Get top queries per URL
            top_queries = get_top_queries_per_url(df)
            
            if len(top_queries) == 0:
                st.warning("No non-branded queries found in the dataset. Please check your branded terms or upload a different dataset.")
                return
            
            # Display summary of URLs and queries being analyzed
            unique_urls = top_queries['Landing Page'].nunique()
            total_queries = len(top_queries)
            st.write(f"Analyzing top queries for {unique_urls} URLs (Total queries: {total_queries})")
            
            # Initialize collections for results and errors
            results = []
            not_found_urls = set()  # Using set to avoid duplicates
            other_errors = []
            
            # Create progress bar with proper chunking
            total_analyses = len(top_queries)
            total_chunks = min(100, total_analyses)  # Limit to 100 chunks
            chunk_size = max(1, total_analyses // total_chunks)
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

            # Set progress to 100% when done
            progress_bar.progress(1.0)

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
