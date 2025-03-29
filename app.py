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

# Job recommendation based on skills, location, and experience
def recommend_jobs(user_skills, user_location, user_experience, top_n=5):
    # Transform user skills into TF-IDF vector
    user_vector = tfidf.transform([user_skills])
    similarities = cosine_similarity(user_vector, job_tfidf_matrix).flatten()
    
    # Add similarity scores to job_df
    job_df["similarity"] = similarities

    # Filter jobs based on experience level
    filtered_jobs = job_df[(job_df["minExp"] <= user_experience) & (job_df["maxExp"] >= user_experience)]

    # Prioritize jobs that match location
    filtered_jobs["location_match"] = filtered_jobs["location"].str.contains(user_location, case=False, na=False).astype(int)
    
    # Sort by location match and similarity score
    filtered_jobs = filtered_jobs.sort_values(by=["location_match", "similarity"], ascending=[False, False])

    # Select top N jobs
    top_jobs = filtered_jobs.head(top_n)
    
    return top_jobs[['jobId', 'jobTitle', 'location', 'minExp', 'maxExp', 'skills']].to_dict(orient='records')

# Home Route
@app.route('/')
def home():
    return render_template("index.html")

# Job Recommendation Route
@app.route('/recommend', methods=["POST"])
def recommend():
    user_name = request.form["user_name"]
    user_location = request.form["user_location"]
    user_experience = int(request.form["user_experience"])
    user_skills = request.form["user_skills"]
    
    recommendations = recommend_jobs(user_skills, user_location, user_experience)

    return render_template("result.html", name=user_name, jobs=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
