from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resume content
# Job description
# TF-IDF based text matching
# Match report
# Suggestion logic

# Match report
resume_text = """
Sabur A
Python developer with 2.5 years experience
Skills: Python, Java, Django,"HTML","CSS", "JAVASCRIPT", "AWS" "AZURE"
Project: Managing Inventory, system-based
Education: B.COM with Python Advanced
Certification: Python full-stack developer - GUVI
"""

job_description = """
We are hiring a Python developer with 2+ years of experience.
Requirement: Strong in Python and Django, experience with database management (MySQL/PostgreSQL),
familiarity with HTML/CSS/JavaScript.
Bonus: Experience with cloud platforms like AWS or Azure, CI/CD pipeline knowledge is a plus.
"""


documents = [resume_text, job_description]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

print("\nResume–Job Match Report:\n")
print(f"🔍 Similarity Score: {similarity:.2f}")

if similarity > 0.6:
    match_level = "✅ Good Match"
elif similarity > 0.4:
    match_level = "🟡 Partial Match"
else:
    match_level = "🔴 Poor Match"

print(f"📌 Match Level: {match_level}\n")


missing_keywords = []
expected_skills = ["HTML", "CSS", "JavaScript", "AWS", "Azure", "CI/CD", "MySQL", "PostgreSQL"]
for skill in expected_skills:
    if skill.lower() not in resume_text.lower():
        missing_keywords.append(skill)

if missing_keywords:
    print("🛠 Suggestions to Improve Resume:")
    print("Add or highlight experience in:", ", ".join(missing_keywords))
else:
    print("🎉 Resume already covers all major job requirements!")
