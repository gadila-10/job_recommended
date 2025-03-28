from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved models and datasets
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/job_tfidf_matrix.pkl", "rb") as f:
    job_tfidf_matrix = pickle.load(f)

with open("models/job_df.pkl", "rb") as f:
    job_df = pickle.load(f)

with open("models/edu_df.pkl", "rb") as f:
    edu_df = pickle.load(f)

# Recommendation Function
def recommend_jobs(applicant_id, top_n=5):
    candidate = edu_df[edu_df['applicantId'] == applicant_id]
    if candidate.empty:
        return []

    candidate = candidate.iloc[0]
    cand_exp = candidate['estExp']

    eligible_jobs = job_df[job_df['avgExp'] <= cand_exp].copy()
    if eligible_jobs.empty:
        return []

    eligible_indices = eligible_jobs.index.tolist()
    eligible_tfidf = job_tfidf_matrix[eligible_indices]

    candidate_vector = np.asarray(eligible_tfidf.mean(axis=0))
    similarities = cosine_similarity(candidate_vector, eligible_tfidf).flatten()
    eligible_jobs['similarity'] = similarities

    top_jobs = eligible_jobs.sort_values(by='similarity', ascending=False).head(top_n)
    return top_jobs[['jobId', 'jobTitle', 'location', 'minExp', 'maxExp', 'skills']].to_dict(orient='records')

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=["POST"])
def recommend():
    applicant_id = request.form["applicant_id"]
    recommendations = recommend_jobs(applicant_id)
    return render_template("result.html", jobs=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
