import re
import html
import string
from typing import Optional
import numpy as np
import emoji

def clean_text_for_analysis(text: str) -> str:
    """
    Clean and preprocess text for mental health sentiment analysis, 
    matching the exact steps used to label the original training data.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # HTML decoding (Matches original)
    text = html.unescape(text)
    
    # 💡 ADDED: Convert emojis to text (Crucial for original labeling accuracy)
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
    except NameError:
        # Fallback if the 'emoji' library is not imported/installed
        pass 
        
    # Remove URLs (Matches original's strict removal)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove mentions (Matches original's removal)
    text = re.sub(r"@\w+", "", text)
    
    # Keep hashtag content, remove # (Matches original)
    text = re.sub(r"#(\w+)", r"\1", text)   
    
    # Normalize whitespace and trim (Matches original)
    text = re.sub(r"\s+", " ", text).strip()
    
    # --- (Removed your custom excessive punctuation/slang rules, as they were not in the original cleaning) ---
    # We prioritize matching the cleaning function used to generate the training data.
    
    # Ensure text isn't too long for model processing
    if len(text) > 500:     
        text = text[:500] + "..."
    
    return text

def extract_emotional_keywords(text: str) -> list:
    """
    Extract potential emotional keywords from text that might indicate mental health concerns.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of emotional keywords found
    """
    # Mental health related keywords (basic set)
    emotional_keywords = {
        'positive': ['happy', 'joy', 'love', 'excited', 'grateful', 'blessed', 'amazing', 'wonderful', 'great'],
        'negative': ['sad', 'depressed', 'anxiety', 'anxious', 'worry', 'fear', 'lonely', 'hopeless', 'worthless'],
        'crisis': ['suicide', 'kill myself', 'end it all', 'not worth living', 'hurt myself', 'self harm', 'dying']
    }
    
    text_lower = text.lower()
    found_keywords = {'positive': [], 'negative': [], 'crisis': []}
    
    for category, keywords in emotional_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords[category].append(keyword)
    
    return found_keywords

def preprocess_for_model_input(text: str, model_type: str = 'bert') -> str:
    """
    Model-specific preprocessing for different transformer models.
    
    Args:
        text (str): Input text
        model_type (str): Type of model ('bert', 'roberta', 'distilbert')
        
    Returns:
        str: Text preprocessed for specific model
    """
    # Common cleaning
    text = clean_text_for_analysis(text)
    
    if model_type.lower() == 'roberta':
        # RoBERTa handles capitalization and punctuation better
        # Less aggressive cleaning
        return text
    
    elif model_type.lower() == 'distilbert':
        # DistilBERT benefits from more standardized input
        text = text.lower()
        return text
    
    else:  # Default BERT processing
        return text

def validate_input_text(text: str) -> tuple[bool, str]:
    """
    Validate input text for processing.
    
    Args:
        text (str): Input text to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text input is required"
    
    if len(text.strip()) < 3:
        return False, "Text is too short for meaningful analysis"
    
    if len(text) > 2000:
        return False, "Text is too long (maximum 2000 characters)"
    
    # Check for potential spam or nonsensical input
    if len(set(text.lower().split())) < 2:
        return False, "Text appears to be repetitive or nonsensical"
    
    return True, ""

def anonymize_personal_info(text: str) -> str:
    """
    Basic anonymization of potential personal information.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with personal info anonymized
    """
    # Remove potential phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Remove potential email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove potential names (basic pattern - capitalize words that might be names)
    # This is a simple approach and might not catch all names
    words = text.split()
    processed_words = []
    
    for word in words:
        # If word is capitalized and not at start of sentence, might be a name
        if (len(word) > 2 and word[0].isupper() and word[1:].islower() and 
            word not in ['The', 'This', 'That', 'There', 'Then', 'They', 'Today', 'Tomorrow']):
            processed_words.append('[NAME]')
        else:
            processed_words.append(word)
    
    return ' '.join(processed_words)

def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about the input text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Text statistics
    """
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'average_word_length': np.mean([len(word) for word in words]) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }
    
    return stats
