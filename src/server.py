#!/usr/bin/env python

from playsound import playsound

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from classifier import classify

import sys
sys.path.append('./src/Grad-TTS/')
from inference import say

app = Flask(__name__)
CORS(app)


@app.route("/")
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def hello_world():
    return "<p>Server reachable!</p>"


@app.route("/synthesize", methods=["POST"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def synthesize():
    if request.method == "POST":
        content = request.json
        sentence = content["sentence"]

        print(f"Processing: {sentence} in progress...")
        lang = classify(sentence)
        print(f"Language: {lang} detected", end="\n\n")
        print(f"Let's synthesize")

        if lang == "EN":
            playsound(say(sentence))
        else:
            playsound(say("Sorry, you have to wait for the french model; only english one available"))

        return jsonify({"speech": "future *.raw file ?"})
    else:
        return jsonify({"speech": "nie dla psa kiełbasa"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
