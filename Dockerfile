# FROM frolvlad/alpine-miniconda3
FROM continuumio/miniconda3

RUN apt update \
	&& apt -y install iputils-ping

COPY ./frontend/ /frontend/

# copy conda, environment configuration files
COPY ./env/ /src/
# copy project source files
COPY ./src/ /src/
WORKDIR /src

SHELL ["/bin/bash", "--login", "-c"]

# recreate conda environment
RUN conda env create -f environment.yml \
	&& rm environment.yml \
	&& echo "conda activate tts-env" >> ~/.bashrc

# sanity-check: verify whether conda loaded libraries correctly
RUN echo "does conda work?" \
	&& python -c "import flask"
