<div align="center" height="130px">
  <img src="./docs/logo.png" alt="Logotype"/><br/>
  <p></p>
</div>

> Express your thoughts — achieve your goals!

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e900ed98717c4c61b7dff288a075c6e8)](https://www.codacy.com/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/branch/main/graph/badge.svg?token=Q21TAQTAZY)](https://codecov.io/gh/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator)
[![CI/CD master](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/ci.yaml/badge.svg)](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/ci.yaml)
[![Kaggle master](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/kaggle.yaml/badge.svg)](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/actions/workflows/kaggle.yaml)

<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#motivation">About the project</a></li>
    <li><a href="getting-started">Getting Started</li>
    <li><a href="#qe">Quality Ensuring</a></li>
    <li><a href="#contribute">How to contribute?</a></li>
    <li><a href="#contributors">Major Contributors</a></li>
  </ul>
</details>

<br>

## 📝 About the project <a name="motivation"></a>

Writing skills are essential for a modern person and must be developed throughout the entire life. Good piece of writing might help in career, relationships, personal effectiveness, and even in self-understanding. However, improving this competency could be problematic in the absence of a reviewer.

Accordingly, creation of an open-source automated text evaluator tends to be a natural step towards enhanced writing skills within society. Firstly, it can speed up the essay review processes done by teachers. Secondly, such a tool can make assessment more unbiased. Thirdly, it might help foreigners to identify linguistic gaps and thereby facilitate the learning process.

As a part of [feedback prize](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) competition our goal is to create an automatic solution that scores students’ essays using multiple criteria: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. For each criterion, the system assigns a score from `1.0` to `5.0`.

## ⚡ Getting Started <a name="getting-started"></a>

### Use our service
Check out our relevant version that is [available on this link](https://huggingface.co/spaces/Booguy/automatic-essay-evaluator)!

### Prerequisites:
1. GNU `make` utility ([link](https://www.gnu.org/software/make/))
2. Python of version 3.7.13 ([link](https://www.python.org/downloads/release/python-3713/))
3. Packaging manager `poetry` ([link](https://python-poetry.org))
4. At least 2Gb on your hard disk

Once all of above is satisfied, execute the following to setup the poetry:
```
poetry lock
poetry --no-root install
```


### Run application locally
To your delight, it's done via a single command:
```
poetry run make build
```

## 🚀 Quality Ensuring <a name="qe"></a>
We put a significant effort to (partially) automate routine operations and restrict programmers from violating style rules and designing non-working code:
- [Using Poetry](.pyproject.toml) to avoid dependency hell (replacement for `pip` package);
- [Continuous integration workflow](.github/workflows/ci.yaml) that performs linting according to [PEP8](.flake8) and [unit testing](tests);
- [Pre-commit hooking](.pre-commit-config.yaml) that runs autopep8, dependencies sorting, and autoflake;
- [Submission workflow](.github/workflows/kaggle.yaml) that loads our best performing solution to Kaggle kernel;
- [Configurable experiments](src/config/conf/) via Hydra that keeps our studies clean and structured;
- [Syncing experiments](src/model_finetuning/train.py) in [Weights & Biases](https://wandb.ai/site) that helps us to monitor progress of our experiments;
- [Automate building of project](Makefile) via Makefile;
- [Evaluation via cross-validation](src/cross_validate.py) that is cosidered to be the most objective amid possible ways to assess generalization of a model;
- [Reproducible experimentation](src/utils.py) that guarantees that same set-up will give equal results on different machines;
- [Notifications in Telegram](src/utils.py) when training is completed;
- Badges with codecov, codacy, continuous integration, and kaggle submission;
- Used [snyk](https://snyk.io) to find vulnerabilities, e.g., in this [PR](https://github.com/Vitsyn-Morgunov-and-Nikulin/automatic-essay-evaluator/pull/21);
- Tried to attach commits to tickets (mostly in latter part of development).

## ✏️ How to contribute? <a name="contribute"></a>
In our development process we followed practices described by Uncle Bob in his magnificent "Clean Code". Please, consult this book in case any trouble.

Make a fork of this repository, and develop your own tool. Make sure it is error-free and the test coverage is at least 60 percent. Update `config` files accordingly, and check their operability.

While producing your code, use this famous [git workflow](https://nvie.com/posts/a-successful-git-branching-model/). Also note that our branches use prefixes `feature/`, `fix/`, and `ci-cd`.

Further, send a pull request. In the comment, write the main features of the tool, the technology stack used, and a brief description of the algorithms. This should be enough for us to accept your code.

> To check the quality of the code, we use `flake8` and `codacy`.

## 💻 Contributors <a name="contributors"></a>
**Shamil Arslanov** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>s.arslanov@innopolis.university</a> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/homomorfism">@homomorfism</a> <br>

**Maxim Faleev** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>m.faleev@innopolis.university</a> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/implausibleDeniability">@implausibleDeniability</a> <br>

**Danis Alukaev** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>d.alukaev@innopolis.university</a> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/DanisAlukaev">@DanisAlukaev</a> <br>

## 📃 Licence <a name="license"></a>
`Lunguask` is free and open-source software licensed under the [MIT License](LICENSE).
