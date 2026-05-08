"""
datasets.py — Dataset registry.

Maps human-readable dataset names to one or more Kaggle slugs.
If a primary slug fails (403 / removed / policy-restricted), the loader
can automatically try the next slug in the list.
"""

DATASETS = {
    "Student Performance to Career Signals": [
        "spscientist/students-performance-in-exams",
        "rabieelkharoua/students-performance-dataset",
    ],
    "Candidate Job Role Recommendation": [
        "ckshetty/candidate-job-role-dataset",
        "hmnshudhmn24/ai-resume-matcher-dataset-2000-samples",
    ],
    "Career Guidance Survey": [
        "adityamadupalli/student-career-recommendation-dataset",
        "thedevastator/jobs-dataset-from-glassdoor",
    ],
    "Academic + Employability Signals": [
        "uciml/student-alcohol-consumption",
        "mylesoneill/student-scores",
    ],
}

VIZ_DATASETS = {
    "Handwriting Math": [
        "ntcuong2103/crohme2019",
        "prajwalchy/hme100k-dataset",
    ],
}