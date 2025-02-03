import streamlit as st
import pandas as pd
import requests
from html.parser import HTMLParser
import re
from typing import List, Dict

class SEOParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.h1s = []
        self.h2s = []
        self.current_tag = None
        self.main_content = []
        self.in_nav = False
        self.in_header = False
        self.in_footer = False
        self.in_sidebar = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag == 'meta' and attrs_dict.get('name') == 'description':
            self.meta_description = attrs_dict.get('content', '')
        elif tag in ['nav', 'header', 'footer', 'aside']:
            setattr(self, f'in_{tag}', True)

    def handle_endtag(self, tag):
        if tag in ['nav', 'header', 'footer', 'aside']:
            setattr(self, f'in_{tag}', False)
        self.current_tag = None

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_nav or self.in_header or self.in_footer or self.in_sidebar:
            return

        if self.current_tag == 'title':
            self.title = data
        elif self.current_tag == 'h1':
            self.h1s.append(data)
        elif self.current_tag == 'h2':
            self.h2s.append(data)
        elif self.current_tag in ['p', 'div', 'span', 'article']:
            self.main_content.append(data)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines"""
    return ' '.join(text.split())

def check_keyword_presence(text: str, keyword: str) -> bool:
    """Check if keyword is present in text (case insensitive)"""
    if not text or not keyword:
        return False
    return keyword.lower() in text.lower()

def analyze_url(url: str, keyword: str) -> Dict:
    """Analyze a URL for SEO elements and keyword presence"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        parser = SEOParser()
        parser.feed(response.text)
        
        # Get first 5 H2s or pad with empty strings
        h2s = parser.h2s[:5]
        h2s.extend([''] * (5 - len(h2s)))
        
        main_content = ' '.join(parser.main_content)
        
        return {
            'title': clean_text(parser.title),
            'title_contains': check_keyword_presence(parser.title, keyword),
            'meta_description': clean_text(parser.meta_description),
            'meta_contains': check_keyword_presence(parser.meta_description, keyword),
            'h1': clean_text(parser.h1s[0] if parser.h1s else ''),
            'h1_contains': check_keyword_presence(parser.h1s[0] if parser.h1s else '', keyword),
            'h2_1': clean_text(h2s[0]),
            'h2_1_contains': check_keyword_presence(h2s[0], keyword),
            'h2_2': clean_text(h2s[1]),
            'h2_2_contains': check_keyword_presence(h2s[1], keyword),
            'h2_3': clean_text(h2s[2]),
            'h2_3_contains': check_keyword_presence(h2s[2], keyword),
            'h2_4': clean_text(h2s[3]),
            'h2_4_contains': check_keyword_presence(h2s[3], keyword),
            'h2_5': clean_text(h2s[4]),
            'h2_5_contains': check_keyword_presence(h2s[4], keyword),
            'main_content': clean_text(main_content[:1000]),  # Limit content length for display
            'content_contains': check_keyword_presence(main_content, keyword)
        }
    except Exception as e:
        st.error(f"Error analyzing {url}: {str(e)}")
        return None

def is_branded_query(query: str, branded_terms: List[str]) -> bool:
    """Check if a query contains any branded terms"""
    query = query.lower()
    return any(brand.lower() in query for brand in branded_terms if brand.strip())

def get_top_queries_per_url(df: pd.DataFrame, max_queries: int = 10) -> pd.DataFrame:
    """Get top queries by clicks for each unique URL"""
    # Remove branded queries if any were marked
    if 'is_branded' in df.columns:
        df = df[~df['is_branded']]
    
    # Sort URLs by total clicks to prioritize more important pages
    url_total_clicks = df.groupby('Landing Page')['Clicks'].sum().sort_values(ascending=False)
    
    results = []
    for url in url_total_clicks.index:
        url_queries = df[df['Landing Page'] == url].copy()
        
        # Get queries with clicks first
        queries_with_clicks = url_queries[url_queries['Clicks'] > 0].nlargest(max_queries, 'Clicks')
        
        # If we have less than max_queries with clicks, add some zero-click queries sorted by impressions
        if len(queries_with_clicks) < max_queries:
            remaining_slots = max_queries - len(queries_with_clicks)
            zero_click_queries = url_queries[url_queries['Clicks'] == 0].nlargest(remaining_slots, 'Impressions')
            queries_for_url = pd.concat([queries_with_clicks, zero_click_queries])
        else:
            queries_for_url = queries_with_clicks
        
        results.append(queries_for_url)
    
    return pd.concat(results)

def format_avg_position(position_str: str) -> float:
    """Format average position to have correct decimal places"""
    try:
        # Remove any commas or dots from the string
        clean_str = str(position_str).replace(',', '').replace('.', '')
        
        # Convert to float
        num = float(clean_str)
        
        # If it's a large number (like 10.035.289.504.639.900), it's probably meant to be 1.0
        if num > 100:
            # Get first digit
            first_digit = int(str(clean_str)[0])
            if first_digit == 1:
                return 1.0
            elif first_digit == 2:
                return 2.0
            # If it starts with 9, it's probably meant to be 9.something
            elif first_digit == 9:
                # Take first 4 characters and format properly
                position = float(clean_str[:4]) / 100
                return round(position, 2)
        
        return round(float(position_str), 2)
    except (ValueError, TypeError):
        return 0.0

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

def main():
    st.set_page_config(page_title="On Page Quick Wins", layout="wide")
    
    st.title("On Page Quick Wins")
    st.write("Upload your GSC performance report to analyze keyword usage in your pages.")
    
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
            
            results = []
            
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
                
                analysis = analyze_url(row['Landing Page'], row['Query'])
                if analysis:
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
                for url in results_df['URL'].unique():
                    url_results = results_df[results_df['URL'] == url]
                    st.write(f"### {url}")
                    st.dataframe(url_results, use_container_width=True)
                
                # Export option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Report",
                    data=csv,
                    file_name="seo_analysis_report.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
