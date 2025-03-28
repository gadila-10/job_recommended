from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved models and datasets
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/job_tfidf_matrix.pkl", "rb") as f:
    job_tfidf_matrix = pickle.load(f)

with open("models/job_df.pkl", "rb") as f:
    job_df = pickle.load(f)

# Job recommendation based on skills
def recommend_jobs(user_skills, top_n=5):
    user_vector = tfidf.transform([user_skills])
    similarities = cosine_similarity(user_vector, job_tfidf_matrix).flatten()
    
    job_df["similarity"] = similarities
    top_jobs = job_df.sort_values(by="similarity", ascending=False).head(top_n)
    
    return top_jobs[['jobId', 'jobTitle', 'location', 'minExp', 'maxExp', 'skills']].to_dict(orient='records')

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=["POST"])
def recommend():
    user_skills = request.form["user_skills"]
    recommendations = recommend_jobs(user_skills)
    return render_template("result.html", jobs=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
