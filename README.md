Unsupervised Topic Modeling & Summarization of Scientific Research Documents
📌 Project Overview
This project is a Streamlit web application that applies Unsupervised Topic Modeling and Summarization techniques to extract meaningful topics and summaries from scientific research documents.

It supports multiple input formats:

📄 PDF
📝 DOCX
📜 TXT
🔗 URL-based text extraction
🔤 Direct text input
The application uses LDA (Latent Dirichlet Allocation) for topic modeling and TextRank-based summarization to generate concise insights from research papers.

🚀 Features
✅ Extracts topics dynamically using LDA + TF-IDF keyword extraction
✅ Summarizes scientific documents into concise paragraphs
✅ Supports multiple file types (PDF, DOCX, TXT, URLs)
✅ Extracts and filters images from research papers
✅ Simple and interactive web UI using Streamlit

🏗️ Tech Stack
Python
Streamlit (Web Framework)
Natural Language Processing (NLP)
nltk, summa
Machine Learning
sklearn, TfidfVectorizer, LatentDirichletAllocation
Document Processing
pymupdf (PDF)
python-docx (DOCX)
Web Scraping
requests, beautifulsoup4
Data Handling
pandas, numpy
