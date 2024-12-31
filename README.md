# BioProcessML

Welcome to **BioProcessML**, a personal journey into the world of bioprocess modeling! This repository showcases my exploration of advanced machine learning techniques, including Gaussian Processes (GPs), Long Short-Term Memory networks (LSTMs), and Recurrent Neural Networks (RNNs), alongside hybrid modeling approaches. 

**Warning:** This journey might take a year or two

## About This Project

In this repository, I aim to develop predictive models for bioprocesses while incorporating professional MLOps practices. Each experiment and code snippet represents a step in my learning journey, emphasizing hands-on experience and practical applications.

## Technologies Used

### Machine Learning Techniques
- **Gaussian Processes (GPs)**
- **Long Short-Term Memory networks (LSTMs)**
- **Recurrent Neural Networks (RNNs)**
- **Others**

### MLOps Tools
- **Logging**: For tracking experiments and model performance.
- **Docker**: For containerizing applications to ensure consistency.
- **GitHub CI/CD**: For continuous integration and deployment of models.
- **MLflow**: For managing the machine learning lifecycle.
- **DVC (Data Version Control)**: For versioning datasets and model files.
- **Apache Airflow**: For orchestrating workflows and automating tasks.


## Resources that are being used to study
- For Deep Learning: [Dive into Deep Learning](https://d2l.ai/)
- For Deep Learning: [Alice’s Adventures in a
differentiable wonderland](https://arxiv.org/abs/2404.17625) (This is condensed resource so need to make sure your basics are clear. I will start with the 'Dive into Deep Learning' and then go through this book to make sure I am not missing anything)
- For Maths (I have forgotten most of them so need some refresher): [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) (I usually revise my concepts each time I come across maths that I can't decipher. This book can serve as a reference. I generally use other web resources tbh, but its to good to have a reference resource.)
- For ML: [PRML](https://www.youtube.com/@sinatootoonian9129)
  
I will have a notebook folder to keep all the notes of the things I have studied. I am following the first book (top-bottom approach). 


## MLOps Workflow
During development, I use DVC and MLflow to manage and track data and model versions. DVC ensures reproducibility by versioning datasets and models, while MLflow tracks experiments, logs metrics, and manages model versions. Once experiments are complete and models are ready, I push all changes to GitHub.

GitHub Actions then automates the CI/CD process, testing, building, and deploying Docker containers for both Airflow DAGs (handling retraining) and the prediction web app. The containers are hosted to simulate a production environment (You can use Azure container registry. I don't have it so I will check everything locally). For the model storage, Google Drive serves as a free storage solution where Airflow uploads the trained models. This allows the web app to access and load the latest models regularly.


## Progress
CNNs (both classical and modern architecture) :heavy_check_mark:
RNNs :heavy_check_mark:
LSTM - In progress
GPs - In progress
