#!/usr/bin/env python

from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin

from classifier import classify

import sys
sys.path.append('./Grad-TTS/')
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
        sentence = " ".join(content["sentence"].split("\n"))

        print(f"Processing: {sentence} in progress...")
        lang = classify(sentence)
        print(f"Language: {lang} detected", end="\n\n")
        print("Let's synthesize")

        out_path = say(sentence, lang)

        return send_file(out_path, mimetype="audio/mpeg", as_attachment=True, download_name=f"{lang}.mp3")
    else:
        out_path = say("Can't touch this", lang)
        return send_file(out_path, mimetype="audio/mpeg", as_attachment=True, download_name="unk.mp3")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
