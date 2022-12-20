FROM python:3.8-slim

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN poetry --no-root install

CMD ["poetry", "run", "make", "build"]
