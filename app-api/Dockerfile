FROM python:3.9.4

WORKDIR /app/src

# Install requirements
ADD ./app-api /app/src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x run.sh

# TODO: First install requirements, then copy the rest of the files.
# TODO: Remove expose
EXPOSE 8001

CMD ["bash", "./run.sh"]