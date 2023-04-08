FROM python:3.9.4

WORKDIR /opt/api

# Install requirements
ADD ./api /opt/api
RUN pip install --upgrade pip
RUN pip install -r /opt/api/requirements.txt

RUN chmod +x /opt/api/run.sh

EXPOSE 8001

CMD ["bash", "./run.sh"]