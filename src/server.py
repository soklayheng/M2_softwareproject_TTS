#!/usr/bin/env python

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin

from classifier import classify

import sys
# sys.path.append('./src/Grad-TTS/')
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
        sentence = content["sentence"]

        print(f"Processing: {sentence} in progress...")
        lang = classify(sentence)
        print(f"Language: {lang} detected", end="\n\n")
        print(f"Let's synthesize")

        if lang == "EN":
            out_path = say(sentence)
            # playsound(say(sentence))
        else:
            default_response = "Sorry, you have to wait for the french model; only english one available"
            out_path = say(default_response)
            # playsound(say(default_response))
        return send_file(out_path, mimetype="audio/wav", as_attachment=True, attachment_filename="sample.wav")
    else:
        return jsonify({"speech": "Can't touch this"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
