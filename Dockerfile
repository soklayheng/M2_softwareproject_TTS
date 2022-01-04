# FROM frolvlad/alpine-miniconda3
FROM continuumio/miniconda3

WORKDIR /src

# copy conda, environment configuration files
COPY ./env/ .

SHELL ["/bin/bash", "--login", "-c"]

# recreate conda environment
RUN conda env create -f environment.yml \
	&& rm environment.yml \
	&& echo "conda activate tts-env" >> ~/.bashrc

# copy project source files
COPY ./src/ .