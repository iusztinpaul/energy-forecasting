FROM python:3.9.4

WORKDIR /opt/frontend/app

ADD ./frontend /opt/frontend
RUN pip install --upgrade pip
RUN pip install -r /opt/frontend/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]