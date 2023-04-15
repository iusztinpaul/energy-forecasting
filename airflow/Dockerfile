FROM apache/airflow:2.5.2

ARG CURRENT_USER=$USER

USER root
# TODO: Clean installed packages
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y python3.9-distutils python3.9-dev build-essential
# RUN apt-get install -y python3.9-dev \
#      build-essential libssl-dev libffi-dev \
#      libxml2-dev libxslt1-dev zlib1g-dev

USER ${CURRENT_USER}
