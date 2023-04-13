FROM python:3.9.4

WORKDIR /app/src

ADD ./app-frontend /app/src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# TODO: First install requirements, then copy the rest of the files.
# TODO: Remove expose
EXPOSE 8501

CMD ["streamlit", "run", "frontend/main.py"]