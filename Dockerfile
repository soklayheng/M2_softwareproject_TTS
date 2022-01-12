# FROM frolvlad/alpine-miniconda3
FROM continuumio/miniconda3

RUN apt-get -y update && apt-get install -y libzbar-dev

WORKDIR /src

# copy conda, environment configuration files
COPY ./env/ .

SHELL ["/bin/bash", "--login", "-c"]

# recreate conda environment
RUN conda env create -f environment.yml \
	&& conda init bash \
	&& conda activate tts-env

# copy language model
COPY ./classifierModel .

# copy project source files
COPY ./src/ .

RUN conda activate tts-env \
	&& cd Grad-TTS/model/monotonic_align \
	&& python setup.py build_ext --inplace \
	&& cd ../../..

EXPOSE 5000

CMD conda activate tts-env \
	&& ./server.py