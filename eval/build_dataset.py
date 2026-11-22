"""Regenerates eval/resumes.json, eval/jds.json and eval/labels.csv.

IMPORTANT: the ratings below are first-pass labels written by the project
author's assistant to bootstrap the evaluation. They are NOT independent human
judgements. Re-rate them yourself (or with a friend) and add harder pairs --
see eval/README.md. Edit this file, run it, then re-run eval/run_eval.py.

Rating scale: 5 = strong fit, 4 = good, 3 = partial / adjacent, 2 = weak, 1 = unrelated.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

RESUMES = {
    "ml_senior": """Summary:
Machine Learning Engineer with 5 years of experience building NLP systems.

Skills:
Python, PyTorch, Hugging Face Transformers, scikit-learn, SQL, Docker, AWS

Experience:
- Fine-tuned BERT and T5 models for text classification and summarization, improving F1 by 14%.
- Built and deployed a real-time inference API with FastAPI and Docker on AWS, serving 50k requests per day.
- Designed evaluation pipelines and A/B tests for NLP models; reduced model latency by 40% with ONNX.
- Mentored 3 junior engineers on model deployment best practices.

Education:
M.Tech in Computer Science""",
    "ml_junior": """Summary:
Final-year student interested in machine learning and NLP.

Skills:
Python, NumPy, Pandas, scikit-learn, Git, basic SQL

Projects:
- Built a sentiment classifier using scikit-learn and TF-IDF on a public reviews dataset.
- Fine-tuned a DistilBERT model with Hugging Face Transformers for a course project.
- Created a Jupyter notebook analysis of housing prices using regression.

Education:
B.Tech in Computer Science, 2026""",
    "data_analyst": """Summary:
Data Analyst with 4 years of experience turning data into business decisions.

Skills:
SQL, Excel, Tableau, Power BI, Python (Pandas), statistics, A/B testing

Experience:
- Built Tableau dashboards used by 30+ stakeholders to track weekly revenue and churn.
- Wrote complex SQL queries over 10M-row tables, cutting report turnaround by 35%.
- Ran A/B tests and presented statistical findings to product and marketing teams.
- Automated monthly reporting with Python and Pandas, saving 12 hours per month.

Education:
B.Sc in Statistics""",
    "frontend": """Summary:
Frontend Engineer with 4 years of experience building responsive web apps.

Skills:
JavaScript, TypeScript, React, Next.js, HTML, CSS, Redux, Jest, Git

Experience:
- Built a React and TypeScript design system used across 5 product teams.
- Improved page load time by 45% with code splitting and Next.js server rendering.
- Wrote unit and integration tests with Jest and React Testing Library, reaching 90% coverage.
- Integrated REST and GraphQL APIs and collaborated with designers on accessibility.

Education:
B.E. in Information Technology""",
    "backend": """Summary:
Backend Engineer with 5 years of experience building scalable services.

Skills:
Python, Django, FastAPI, PostgreSQL, Redis, Docker, AWS, REST APIs, Git

Experience:
- Designed and built REST APIs in Django and FastAPI serving 2M requests per day.
- Optimized PostgreSQL queries and indexes, reducing p95 latency by 50%.
- Containerized services with Docker and deployed them to AWS ECS with CI pipelines.
- Implemented caching with Redis and background jobs with Celery.

Education:
B.Tech in Computer Science""",
    "devops": """Summary:
DevOps Engineer with 6 years of experience in cloud infrastructure and automation.

Skills:
Kubernetes, Docker, Terraform, AWS, Jenkins, GitHub Actions, Prometheus, Grafana, Linux, Bash

Experience:
- Managed production Kubernetes clusters running 200+ microservices on AWS EKS.
- Provisioned infrastructure as code with Terraform, cutting environment setup from days to 30 minutes.
- Built CI/CD pipelines in Jenkins and GitHub Actions, reducing deployment time by 60%.
- Set up monitoring and alerting with Prometheus and Grafana, improving uptime to 99.95%.

Education:
B.Tech in Electronics""",
    "data_eng": """Summary:
Data Engineer with 5 years of experience building batch and streaming pipelines.

Skills:
Python, SQL, Apache Spark, Airflow, Kafka, Redshift, AWS, dbt, Docker

Experience:
- Built Spark pipelines processing 2TB of data daily into a Redshift data warehouse.
- Orchestrated 80+ ETL workflows with Airflow and added data quality checks with dbt.
- Designed a Kafka streaming ingestion layer handling 100k events per second.
- Reduced warehouse costs by 30% through partitioning and query optimization.

Education:
M.Sc in Computer Science""",
    "unrelated": """Summary:
Retail Store Manager with 8 years of experience in customer service and team leadership.

Skills:
Team management, inventory control, customer relations, scheduling, merchandising

Experience:
- Managed a team of 25 associates and increased store sales by 18% year over year.
- Handled inventory planning and reduced stock losses by 12%.
- Trained new hires and resolved customer complaints to maintain a 4.8 satisfaction rating.

Education:
B.Com""",
    "fullstack": """Summary:
Full-stack Developer with 3 years of experience shipping web products.

Skills:
JavaScript, React, Node.js, Express, PostgreSQL, Docker, REST APIs, Git

Experience:
- Built a React dashboard and a Node.js and Express backend for a logistics startup.
- Designed PostgreSQL schemas and REST endpoints serving 20k daily users.
- Containerized the application with Docker and set up basic CI with GitHub Actions.
- Added unit tests and improved front-end performance by 30%.

Education:
B.Tech in Computer Science""",
}

JDS = {
    "ml_engineer": """Job Title: Machine Learning Engineer (NLP)

Requirements:
- Strong experience in Python and deep learning frameworks such as PyTorch
- Hands-on experience with NLP and Transformer models
- Experience fine-tuning and evaluating language models
- Experience deploying ML models to production
- Knowledge of SQL and data processing""",
    "data_analyst": """Job Title: Data Analyst

Requirements:
- Strong SQL skills for querying large datasets
- Experience building dashboards in Tableau or Power BI
- Solid understanding of statistics and A/B testing
- Ability to present insights to non-technical stakeholders
- Proficiency in Excel; Python is a plus""",
    "frontend": """Job Title: Frontend Engineer

Requirements:
- Strong experience with React and TypeScript
- Solid HTML and CSS skills and a focus on accessibility and performance
- Experience writing tests with Jest or similar
- Experience integrating REST or GraphQL APIs
- Familiarity with Git and modern build tooling""",
    "backend": """Job Title: Backend Engineer

Requirements:
- Strong Python experience with Django or FastAPI
- Experience designing and building REST APIs
- Solid knowledge of PostgreSQL and database optimization
- Experience with Docker and deploying services on AWS
- Familiarity with caching and background job processing""",
    "devops": """Job Title: DevOps Engineer

Requirements:
- Hands-on experience with Kubernetes and Docker in production
- Infrastructure as code with Terraform
- Experience building CI/CD pipelines
- Strong knowledge of AWS and Linux
- Experience with monitoring tools such as Prometheus and Grafana""",
    "data_engineer": """Job Title: Data Engineer

Requirements:
- Strong Python and SQL skills
- Experience building data pipelines with Apache Spark and Airflow
- Experience with a cloud data warehouse such as Redshift or Snowflake
- Knowledge of streaming systems such as Kafka
- Focus on data quality and cost optimization""",
}

JD_ORDER = ["ml_engineer", "data_analyst", "frontend", "backend", "devops", "data_engineer"]
# resume -> rating against each JD in JD_ORDER
RATINGS = {
    "ml_senior": [5, 3, 1, 3, 2, 3],
    "ml_junior": [3, 3, 1, 2, 1, 2],
    "data_analyst": [2, 5, 1, 1, 1, 3],
    "frontend": [1, 1, 5, 2, 1, 1],
    "backend": [2, 2, 2, 5, 3, 3],
    "devops": [1, 1, 1, 3, 5, 2],
    "data_eng": [3, 4, 1, 3, 2, 5],
    "unrelated": [1, 1, 1, 1, 1, 1],
    "fullstack": [1, 1, 4, 4, 2, 2],
}


def main():
    with open(os.path.join(HERE, "resumes.json"), "w", encoding="utf-8") as f:
        json.dump(RESUMES, f, indent=1)
    with open(os.path.join(HERE, "jds.json"), "w", encoding="utf-8") as f:
        json.dump(JDS, f, indent=1)
    with open(os.path.join(HERE, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["resume_id", "jd_id", "rating"])
        for resume_id, row in RATINGS.items():
            for jd_id, rating in zip(JD_ORDER, row):
                w.writerow([resume_id, jd_id, rating])
    print(f"wrote {sum(len(r) for r in RATINGS.values())} labelled pairs")


if __name__ == "__main__":
    main()
