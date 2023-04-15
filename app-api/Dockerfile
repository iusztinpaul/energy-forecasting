FROM python:3.9.4

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.4.2

# Configuring poetry.
RUN poetry config virtualenvs.create false

# First copy & install requirements to speed up the build process in case only the code changes.
COPY ./app-api/pyproject.toml /app/src/
COPY ./app-api/poetry.lock /app/src/

RUN poetry install --no-interaction --no-root -vvv

# Copy the rest of the files.
ADD ./app-api /app/src

# Give access to run the run.sh script.
RUN chmod +x run.sh

CMD ["bash", "./run.sh"]
