from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import re
import numpy as np

app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATASET_PATH = os.path.join(BASE_DIR, 'news_dataset.csv')

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

FAKE_KEYWORDS = ["shocking", "unbelievable", "clickbait", "hoax", "rumor"]

def is_fake_news(text):
    """Check if input text contains fake news keywords."""
    text = text.lower()
    for keyword in FAKE_KEYWORDS:
        if keyword in text:
            return "❌ Fake News"
    return "✅ Real News"

def get_word_frequency(texts):
    """Get word frequency from texts."""
    words = []
    for text in texts:
        # Convert to lowercase and split into words
        text_words = re.findall(r'\w+', text.lower())
        words.extend(text_words)
    
    # Remove common words and get top 10
    common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
    word_freq = Counter(word for word in words if word not in common_words and len(word) > 2)
    return dict(word_freq.most_common(10))

def generate_visualizations():
    """Generate and save visualizations."""
    try:
        # Read the dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Clear any existing plots
        plt.close('all')
        
        # 1. News Distribution Chart
        plt.figure(figsize=(10, 6))
        news_counts = df["label"].value_counts()
        bars = news_counts.plot(kind="bar", color=["red", "green"])
        plt.title("Distribution of Fake vs Real News", pad=20, fontsize=14)
        plt.xlabel("News Category", fontsize=12)
        plt.ylabel("Number of Articles", fontsize=12)
        plt.xticks(rotation=0)
        
        # Add value labels
        for i, v in enumerate(news_counts):
            bars.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "distribution_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Word Frequency Analysis
        plt.figure(figsize=(12, 6))
        word_freq = get_word_frequency(df['text'])
        plt.bar(word_freq.keys(), word_freq.values(), color='skyblue')
        plt.title("Top 10 Most Common Words", pad=20, fontsize=14)
        plt.xlabel("Words", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "word_frequency.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Text Length Distribution (Box Plot)
        df['text_length'] = df['text'].str.len()
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='label', y='text_length', data=df, palette=['red', 'green'])
        plt.title("Text Length Distribution by News Type", pad=20, fontsize=14)
        plt.xlabel("News Category", fontsize=12)
        plt.ylabel("Text Length (characters)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "length_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Word Count vs Text Length Scatter Plot
        df['word_count'] = df['text'].str.split().str.len()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='text_length', y='word_count', hue='label', alpha=0.6)
        plt.title("Word Count vs Text Length", pad=20, fontsize=14)
        plt.xlabel("Text Length (characters)", fontsize=12)
        plt.ylabel("Word Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "scatter_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Average Word Length Distribution
        df['avg_word_length'] = df['text_length'] / df['word_count']
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='avg_word_length', hue='label', bins=30, alpha=0.6)
        plt.title("Average Word Length Distribution", pad=20, fontsize=14)
        plt.xlabel("Average Word Length (characters)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "word_length_dist.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate statistics
        stats = {
            'total_articles': len(df),
            'fake_count': len(df[df['label'] == 'Fake']),
            'real_count': len(df[df['label'] == 'Real']),
            'fake_percentage': round(len(df[df['label'] == 'Fake']) / len(df) * 100, 2),
            'real_percentage': round(len(df[df['label'] == 'Real']) / len(df) * 100, 2),
            'avg_fake_length': round(df[df['label'] == 'Fake']['text_length'].mean(), 2),
            'avg_real_length': round(df[df['label'] == 'Real']['text_length'].mean(), 2),
            'avg_fake_words': round(df[df['label'] == 'Fake']['word_count'].mean(), 2),
            'avg_real_words': round(df[df['label'] == 'Real']['word_count'].mean(), 2)
        }

        return stats
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return None

# Generate visualizations at startup
STATS = generate_visualizations()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Detect fake news from user input."""
    data = request.json
    result = is_fake_news(data["news"])
    return jsonify({"result": result})

@app.route("/dataset-visual")
def show_dataset_visual():
    """Show the dataset analysis page."""
    if STATS is None:
        return render_template("visualization.html", error="Error loading visualization")
    
    return render_template("visualization.html", 
                         stats=STATS,
                         images={
                             'distribution': "distribution_chart.png",
                             'word_freq': "word_frequency.png",
                             'length_boxplot': "length_boxplot.png",
                             'scatter_plot': "scatter_plot.png",
                             'word_length_dist': "word_length_dist.png"
                         })

if __name__ == "__main__":
    app.run(debug=True)