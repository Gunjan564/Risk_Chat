# Mental Health Support Assistant

## Overview

This is a mental health support chatbot application that uses ensemble machine learning models to assess mental health risk levels from user text input. The system employs multiple pre-trained transformer models (BERT, RoBERTa, DistilBERT) to analyze sentiment and classify mental health risk into categories (low, moderate, high, no risk). Built with Streamlit for an accessible web interface, the application provides real-time conversational support while performing text-based mental health risk assessment.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Design Pattern**: Single-page application with chat interface
- **UI Components**: Custom CSS styling for mental health-themed interface with chat containers, user/bot message bubbles
- **Layout**: Wide layout with collapsed sidebar for focused user experience
- **State Management**: Streamlit's session state for conversation history

### Backend Architecture
- **Core Framework**: Python-based with PyTorch for deep learning inference
- **Model Architecture**: Ensemble learning approach combining multiple transformer models
  - **Rationale**: Ensemble methods reduce individual model bias and improve prediction reliability for sensitive mental health assessments
  - **Components**: BERT, RoBERTa, and DistilBERT models working in concert
  - **Inference Mode**: Models run in evaluation mode (`.eval()`) for production inference
- **Text Processing Pipeline**: Multi-stage preprocessing optimized for emotional context preservation
  - HTML decoding
  - URL/mention normalization
  - Slang expansion while maintaining emotional indicators
  - Punctuation normalization to preserve sentiment signals
- **Risk Classification**: Four-level classification system (low, moderate, high, no risk)
- **Device Management**: Automatic CUDA/CPU detection for flexible deployment

### Data Storage Solutions
- **Model Storage**: Pre-trained models loaded from HuggingFace model hub
  - `cardiffnlp/twitter-roberta-base-sentiment` for BERT and RoBERTa
  - `distilbert-base-uncased-finetuned-sst-2-english` for DistilBERT
- **Session Data**: In-memory storage via Streamlit session state (no persistent database)
- **Conversation History**: Temporary storage during user session only

### Authentication and Authorization
- **Current State**: No authentication implemented
- **Access Model**: Public access to the web application
- **Rationale**: Mental health support tool designed for accessibility; authentication could create barriers to seeking help

## External Dependencies

### Machine Learning Libraries
- **PyTorch** (v2.1.2+): Deep learning framework for model inference
- **Transformers** (v4.39.2+): HuggingFace library for pre-trained transformer models
- **scikit-learn**: Ensemble voting classifier and model evaluation utilities

### Data Processing
- **NumPy** (v1.25.2): Numerical computing for tensor operations
- **Pandas**: Data manipulation and conversation logging

### Pre-trained Models (HuggingFace Hub)
- **cardiffnlp/twitter-roberta-base-sentiment**: Sentiment analysis fine-tuned on Twitter data
- **distilbert-base-uncased-finetuned-sst-2-english**: Lightweight sentiment classification model

### Web Framework
- **Streamlit**: Web application framework for rapid deployment and interactive UI

### Development Environment
- **Google Colab Integration**: Training/experimentation environment (evidenced by attached notebook)
- **Kaggle API**: Dataset access for model training
- **Accelerate** (v0.28.0): Distributed training and mixed precision support

### Third-party Services
- **HuggingFace Model Hub**: Model download and versioning
- **Google Drive**: Model storage and collaboration (for training phase)

### Optional Dependencies
- **CUDA Toolkit**: GPU acceleration when available
- **Gensim**: Word embeddings and topic modeling (referenced in training code)