# M2_softwareproject_TTS
Software Project

## Manual
Manual server run may require some modules/paths tweeking; therefore, dockerized option is recommended.
Previous development stage (without dockerization) used the command:
```sh
$ python ./src/server
```

To run web app `cd` to `frontend` and run:
```sh
$ npm start
```

Now You can open up the app in the browser and play with the tool.

## Auto
To build server image run command:
```sh
$ docker build -t tts-minicuda .
```

Once finished, you can run container by running:
```sh
docker run -P tts-minicuda:latest
```

In this stage, server can take requests and provide responses. E.g. use your favourite tool for testing APIs and send `POST` request to: `172.17.0.2:5000/synthesize` with sample payload:
```json
{
    "sentence": "Parles-tu français ?"
}
```

As current app's version only supports English, server returns synthesized response in English informing about the problem. In case English sentence was provided, you should hear its sythesized version back.

Ultimately, once frontend is more ready, frontend-backend communication will be established through private network in docker container and only a single port will be exposed externally to allow connection with app's ui. A single `docker-compose up -d` will be enough to spin up the whole app.