import re
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['ExtraDB']

# Liste des noms de collections
collection_names = ['articles_F24', 'articles_Liberation', 'articles_Jeune_Afrique', 'articles_Lefigaro', 'articles_SeneNews']

def clean_text(text):
    # Supprimer les balises HTML
    cleaned_text = BeautifulSoup(text, 'html.parser').get_text()

    # Convertir en minuscules
    cleaned_text = cleaned_text.lower()

    # Supprimer la ponctuation et les caractères spéciaux
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    return cleaned_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    results = search_in_database(query)
    return render_template('resultats.html', results=results)

def search_in_database(query):
    all_results = []

    for collection_name in collection_names:
        collection = db[collection_name]
        for article in collection.find():
            title = article.get('title', '')
            category = article.get('category', '')
            content = article.get('content', '')
            comments = ' '.join(article.get('comments', []))
            if title or category or content or comments:
                corpus_item = ' '.join([clean_text(title), clean_text(category), clean_text(content), clean_text(comments)])
                all_results.append(corpus_item)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_results)
        query_vector = vectorizer.transform([clean_text(query)])
        similarity_scores = np.dot(tfidf_matrix, query_vector.T).toarray()
        indices = np.argsort(similarity_scores, axis=0)[::-1].flatten()
        results = [all_results[i] for i in indices]

    return results

if __name__ == '__main__':
    app.run(debug=True)
