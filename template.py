from pathlib import Path

project_name = "bioprocess_mlops"
list_of_files = [
    ".github/workflows/.gitkeep",
    "config/logging.dev.ini",
    "config/logging.prod.ini",
    "config/config.yaml",
    "logs/",
    "airflow/dags/training_pipeline.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/main.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/utils/utils.py",
    f"src/{project_name}/utils/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "requirements.txt",
    "requirements_dev.txt",
    ".env",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiment/experiments.ipynb",
    "dvc.yaml",
    "params.yaml",
    "docker-compose.yaml",
    ".flake8"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        filedir.mkdir(parents=True,
                      exist_ok=True)

    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        filepath.touch()
