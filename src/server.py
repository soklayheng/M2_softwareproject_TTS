from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from classifier import classify

app = Flask(__name__)
CORS(app)


@app.route("/")
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def hello_world():
    return "<p>Server reachable!</p>"


@app.route("/synthesize", methods=["GET"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def synthesize():
    if request.method == "GET":
        sentence = request.args.get("sentence")
        # content = request.json
        # sentence = content["sentence"]

        print(f"Processing: {sentence} in progress...")
        lang = classify(sentence)
        print(f"Language: {lang} detected", end="\n\n")
        print(f"Let's synthesize")

        return jsonify({"speech": "future *.raw file ?"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
