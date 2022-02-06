import React, { Component } from "react";

export default class Home extends Component {

  constructor(props) {
    super(props)
    this.state = {
      TextInputValueHolder: '',
      apiresponse: ''
    }
    this.listen = this.listen.bind(this)

  }

  listen = () => {
    var myHeaders = new Headers();
    myHeaders.append("Accept", "application/json");
    myHeaders.append("Content-Type", "application/json");

    var raw = JSON.stringify({
      "sentence": document.getElementById('speechtextid').value
    });

    fetch("/synthesize") // , requestOptions)
      .then(response => response.blob())
      .then(result => {
        const url = URL.createObjectURL(result);
        const audio = new Audio(url);
        audio.play();
        this.setState(state => { state.apiresponse = audio });
      })
      .catch(_ => console.log('error'));
  };

  render() {
    return (
      <div>
        <div class="row">
          <div class="col-xs-6"></div>
          <div class="container bcontent">
            {/* <img className="logo" src="./public/logo.png" /> */}
            <h1>HearOut</h1>

            <hr />
            <div class="row row-grid align-items-center" id="area" rows="60" column="80" />
            <div class="card">

              <div class="card-header">Generate Text to Speech</div>
              <div class="card-body">
                <div>

                  <textarea id={'speechtextid'} class="story" name="story" placeholder="Type Here (Maximum 500 Characters)" rows="9" cols="78" maxLength="500">
                  </textarea>
                  <div><h1>
                    {this.state.apiresponse.speech}
                  </h1></div>

                  <div class="container">
                    <button type="button" onClick={this.listen} class="btn btn-default" style={{ background: "#40E0D0" }}>Listen!</button>
                    <button type="button" class="btn btn-default" style={{ background: "#F08080" }}>Reset</button>
                  </div>
                </div>
              </div>
            </div>

          </div>



          <div class="container-fluid " style={{ background: "#DCDCDC" }}>
            <h3>About Us</h3>
            {/* <img src="logo.png" class="rounded-circle" alt="Logo" width="80" height="100" /> */}
            <h5>Text To Speech delivers human-like, personalized, and engaging user experiencem You now have the ability to translate text into a voice in any language through our Free Online Text to Speech tool. Below you will find some of the languages ​​currently supported on our TTS converter app: English and French</h5>
          </div>
        </div>
      </div>
    )
  }
}
