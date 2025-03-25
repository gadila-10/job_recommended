import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model components
job_tfidf_matrix = pickle.load(open("job_tfidf_matrix.pkl", "rb"))
job_df = pickle.load(open("job_df.pkl", "rb"))
edu_df = pickle.load(open("edu_df.pkl", "rb"))
cosine = pickle.load(open("top_jobs.pkl", "rb"))


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

    candidate_vector = eligible_tfidf.mean(axis=0)
    candidate_vector = np.asarray(candidate_vector)

    similarities = cosine_similarity(candidate_vector, eligible_tfidf).flatten()
    eligible_jobs['similarity'] = similarities

    top_jobs = eligible_jobs.sort_values(by='similarity', ascending=False).head(top_n)
    return top_jobs[['jobId', 'jobTitle', 'location', 'minExp', 'maxExp', 'skills', 'similarity']].to_dict(orient="records")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    applicant_id = int(request.form["applicant_id"])
    recommendations = recommend_jobs(applicant_id)
    return render_template("recommend.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
