FROM python:3.8-slim

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="${PATH}:/root/.poetry/bin"

RUN poetry install

CMD ["$HOME/.poetry/bin/poetry", "run", "make", "build"]
