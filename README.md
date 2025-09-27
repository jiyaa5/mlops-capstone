# MLOps Capstone Starter

This repository is the starter package for the **End-to-End MLOps Assignment**.

## 📌 Steps for Students

1. **Set up environment**
   - Create and activate a virtual environment.
   - Install requirements: `pip install -r requirements.txt`.

2. **Track dataset with DVC**
   - Initialize DVC in repo: `dvc init`
   - Add dataset under `data/housing.csv`: `dvc add data/housing.csv`
   - Commit and push DVC metadata.

3. **Train and log model with MLflow**
   - Run training script: `python src/train.py`
   - Access MLflow UI: `mlflow ui`

4. **Serve model with FastAPI**
   - Run: `uvicorn app:app --reload`
   - Test endpoint at: `http://127.0.0.1:8000/predict`

5. **Containerize with Docker**
   - Build image: `docker build -t mlops-capstone .`
   - Run container: `docker run -p 8000:8000 mlops-capstone`

6. **CI/CD with GitHub Actions**
   - Workflow is defined in `.github/workflows/ci.yml`
   - Push to GitHub → triggers tests and container build.

7. **Infrastructure with Terraform**
   - Use `terraform/main.tf` to provision infra (example: EC2 instance).
   - `terraform init && terraform apply`

8. **Monitoring**
   - `monitor.py` contains hooks for drift detection and alerts.
   - Extend to integrate with email/Slack.

---

## Deliverables

- All pipelines working end-to-end.
- DVC-tracked data and MLflow model logged.
- Running FastAPI containerized API.
- CI/CD pipeline visible in GitHub Actions.
- Terraform provisioned infra resources.
- Monitoring script demo.

Happy coding 🚀
