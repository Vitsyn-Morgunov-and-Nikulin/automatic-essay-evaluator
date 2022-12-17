# Automated Essay Evaluator

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e900ed98717c4c61b7dff288a075c6e8)](https://www.codacy.com/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/branch/mlops/hydra/graph/badge.svg?token=Q21TAQTAZY)](https://codecov.io/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator)
[![CI/CD master](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/ci.yaml/badge.svg)](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/ci.yaml)

**Authors:** Shamil Arslanov, Maxim Faleev, Danis Alukaev
**Group:** B19-DS-01, Innopolis University

## 📝 Motivation

How important it is to write well? In our opinion, it is an essential skill that must be developed throughout your entire life. Good writing helps you in your career, relationships, personal effectiveness, and even in self-understanding.

While practicing writing skills, students generally ask their English teachers to assess texts. But what do people who do not have such an opportunity do? We believe we can help.

## ⚡ What will we implement?

Our goal is to create an open-source API for text quality evaluation. For our project, we extensively use quality ensuring approaches and basic MLOps techniques.

As a team, we take part in the Kaggle competition which requires flexibility in terms of methods used to approach the problem. Most likely we will start with simple feature extraction via Natural Language Toolkit (NLTK) with gradient boosting and end up with state-of-the-art algorithms based on Transformers (e.g. DeBERTa).

## 🐳 Data

We will use the ELLIPSE corpus provided by Vanderbilt University and The Learning Agency Lab as a part of [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) competition. This dataset comprises `3911` argumentative essays each evaluated according to six measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions (see the fugure). These are numbers from `1.0` to `5.0` and step size of `0.5` with larger values corresponding to better proficiency in this aspect.

![Figure 1. Training sample](docs/Screenshot_2022-09-13_at_15.12.27.png)

## 👤 Target audience

Our solution will help teachers speed up the review of students' essays. Also, it will help foreigners to identify gaps in learning English, e.g. if identifies that the essay has poor grammar, then the student needs to be given more exercise in the field, and the same with vocabulary, syntax, etc.

## ✏️ What is expected from the resulting model

Our model scores students’ essays using multiple criteria: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. For each criterion, our model assigns a score from `1.0` to `5.0`. Thereby, our product behaves as an unbiased teacher that checks students’ work and give the grade for each aspect of language knowledge. Objective grading of a student’s essay could identify the sphere of learning English in which the student is struggling the most, indicating the need for a personal study plan change.

## 🚀 Quality Ensuring
We put a significant effort to (partially) automate routine operations and restrict programmers from violating style rules and designing non-working code:
- [Continuous integration workflow](.github/workflows/ci.yaml) that performs linting according to [PEP8](.flake8) and [unit testing](tests);
- [Pre-commit hooking](.pre-commit-config.yaml) that runs autopep8, dependencies sorting, and autoflake;
- [Submission workflow](.github/workflows/kaggle.yaml) that loads our best performing solution to Kaggle kernel;
- [Configurable experiments](src/config/conf/) via Hydra that keeps our studies clean and structured;
- [Syncing experiments](src/model_finetuning/train.py) in [Weights & Biases](https://wandb.ai/site) that helps us to monitor progress of our experiments;
- [Automate building of dataset](Makefile) via Makefile;
- [Evaluation via cross-validation](src/cross_validate.py) that is cosidered to be the most objective amid possible ways to assess generalization of a model;
- [Reproducible experimentation](src/utils.py) that guarantees that same set-up will give equal results on different machines;
- [Notifications in Telegram](src/utils.py) when training is completed;
- Badges with codecov, codacy, and continuous integration.

