import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import string

# Download required NLTK data
def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords', 
        'wordnet': 'corpora/wordnet',
    }
    
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading {resource_name}...")
            nltk.download(resource_name)

download_nltk_resources()

class SocialMediaPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            max_features=3000, 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Extended stopwords for social media and common words
        self.social_media_stopwords = {
            'rt', 'http', 'https', 'com', 'www', 'html', 'like', 'get', 'would',
            'know', 'one', 'see', 'really', 'go', 'think', 'good', 'thank',
            'today', 'day', 'work', 'working', 'project', 'office', 'productive',
            'time', 'people', 'make', 'way', 'look', 'want', 'need', 'use',
            'great', 'first', 'new', 'also', 'back', 'well', 'still', 'much'
        }
        self.stop_words.update(self.social_media_stopwords)
        
        # Disaster context words that should get higher weight
        self.disaster_keywords = {
            'earthquake', 'quake', 'shaking', 'tremor', 'seismic',
            'flood', 'flooding', 'water', 'rain', 'storm', 'hurricane', 
            'wildfire', 'fire', 'burning', 'blaze', 'smoke',
            'drought', 'dry', 'water shortage', 'arid',
            'tornado', 'cyclone', 'typhoon', 'wind',
            'volcano', 'eruption', 'lava', 'ash',
            'emergency', 'disaster', 'crisis', 'evacuate', 'warning',
            'destroyed', 'damage', 'casualty', 'victim', 'rescue'
        }
        
    def clean_text(self, text):
        """Clean and preprocess social media text with better context preservation"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep the content
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep basic sentence structure
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def enhance_disaster_context(self, text):
        """Add weight to disaster-related terms"""
        words = text.split()
        enhanced_words = []
        
        for word in words:
            if word in self.disaster_keywords:
                # Repeat disaster keywords to give them more weight
                enhanced_words.extend([word] * 2)
            else:
                enhanced_words.append(word)
                
        return ' '.join(enhanced_words)
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text with context preservation"""
        if not text:
            return ""
            
        # Enhance disaster context first
        text = self.enhance_disaster_context(text)
            
        tokens = word_tokenize(text)
        
        # Remove stopwords, short tokens, and lemmatize
        processed_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                token not in string.punctuation and
                not token.isdigit()):
                
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)
    
    def preprocess_data(self, df, text_column='Tweets', label_column='Disaster'):
        """Preprocess the entire dataset with better balancing"""
        print("Step 1: Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 10].copy()  # Minimum 10 characters
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} empty/short texts after cleaning")
        
        print("Step 2: Tokenizing and lemmatizing...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Remove empty processed texts
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 5].copy()  # Minimum 5 words
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} empty texts after processing")
        
        # Handle labels if present
        if label_column in df.columns:
            print("Step 3: Encoding labels...")
            # Remove any NaN labels
            df = df.dropna(subset=[label_column])
            
            # Balance the dataset - ensure we have enough non-disaster examples
            disaster_counts = df[label_column].value_counts()
            print(f"Label distribution before balancing: {disaster_counts.to_dict()}")
            
            # If we have a 'non-disaster' class, make sure it's well represented
            non_disaster_labels = ['non-disaster', 'normal', 'none', 'no disaster']
            has_non_disaster = any(label in non_disaster_labels for label in df[label_column].unique())
            
            if not has_non_disaster:
                print("⚠️ No non-disaster class found. Model may have high false positives.")
            
            df['encoded_labels'] = self.label_encoder.fit_transform(df[label_column])
            self.is_fitted = True
            
            print(f"Label classes: {list(self.label_encoder.classes_)}")
            print(f"Final label distribution:")
            print(df[label_column].value_counts())
        
        print(f"Final dataset size: {len(df)}")
        return df
    
    def vectorize_text(self, texts, fit=False):
        """Convert text to TF-IDF vectors with better parameters"""
        if fit:
            X = self.vectorizer.fit_transform(texts)
            print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        else:
            X = self.vectorizer.transform(texts)
        return X
    
    def preprocess_single_text(self, text):
        """Preprocess a single text for prediction"""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        vectorized = self.vectorize_text([processed])
        return vectorized, processed