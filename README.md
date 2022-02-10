# Multilingual Text-to-Speech with Grad-TTS
> UE 805 - Software Project

## Description
Automatic speech synthesis for English and French languages. Dockerized application with published images on Docker Hub. Currently, a project in the v1.0 version.

## Automatic running
The application is fully containerized using `Docker`. We distinguish three microservices:
- `tts-api` <a href="https://hub.docker.com/repository/docker/amillert/tts-api">image URL</a>,
- `tts-ui` <a href="https://hub.docker.com/repository/docker/amillert/tts-ui">image URL</a>,
- `tts-proxy` <a href="https://hub.docker.com/repository/docker/amillert/tts-proxy">image URL</a>.

To pull a specific `Docker` image, one can run a shell command:
``` sh
$ docker pull amillert/tts-api
```

One can utilize images on their own in `docker-compose.yml` file, e.g.:
```
services:
  service-name:
    image: amillert/tts-api:latest
```

Specific containers can be built given image in the `Dockerfile` from its directory, for instance by running:
``` sh
$ docker build -t tts-api .
```

The built container can be then run, e.g.:
``` sh
$ docker run --rm tts-api:latest
```

However, running a container alone is not the most useful. Instead, we encourage one to build and spin up microservices in detached mode through `docker-compose`:
``` sh
$ docker-compose up --build -d
```

## **Deprecated** - manual running
The currect v1.0 version doesn't officially support manual project build, running, as there's no need for it anymore.

## Supported functionality
Once the application is built and running (see above for the instructions), one can reach graphical UI on `localhost`, the website is available through a default `HTTP` port `80`. Reverse-proxy is available through `Nginx` which supports both `React` UI and `Flask` web API server.

The server exposes `POST /synthesis` endpoint, accepting request in `JSON` format - a sentence either in French or English, e.g.:
``` json
{
    "sentence": "Parles-tu français ?"
}
```

Server utilizes <a href="https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">Grad-TTS</a>, which is embedded as a submodule. It was adapted to support both languages for automatic speech synthesis task. It provides the client with an `mp3` file as a response.

The client fetches the data and plays it back in the user's browser.
