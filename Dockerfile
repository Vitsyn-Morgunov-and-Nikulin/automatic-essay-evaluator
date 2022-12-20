FROM python:3.8-slim

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN source $HOME/.poetry/env && poetry update && poetry install

CMD ["poetry", "run", "make", "build"]
