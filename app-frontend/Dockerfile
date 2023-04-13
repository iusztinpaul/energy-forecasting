FROM python:3.9.4

WORKDIR /app/src

ADD ./app-frontend /app/src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend/main.py"]