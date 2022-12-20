FROM python:3.8-slim

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN $HOME/.poetry/bin/poetry install

CMD ["$HOME/.poetry/bin/poetry", "run", "make", "build"]
