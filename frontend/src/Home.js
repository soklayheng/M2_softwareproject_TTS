import React, {Component } from "react";
import { sentence } from "./data";


// export async function fetchVoice()
// { 
//   console.log("calling api");
//     return fetch("http://localhost:5000/synthesize");
//   // return fetch("https://jsonplaceholder.typicode.com/todos/1");
//   // return fetchVoice(sentence)

// }
export default class Home extends Component {

 
  constructor(props) {
    super(props)
    this.state = {
      TextInputValueHolder: '',
      apiresponse:''
    }
    this.listen = this.listen.bind(this)
 
  }
  // listen = () =>{
  //   const { TextInputValueHolder }  = this.state ;
  //     Alert.alert(TextInputValueHolder)
  // }
    


     
listen=()=>{
var myHeaders = new Headers();
myHeaders.append("Accept", "application/json");
myHeaders.append("Content-Type", "application/json");

var raw = JSON.stringify({
  "sentence": document.getElementById('speechtextid').value 
});

var requestOptions = {
  method: 'POST',
  // mode: 'no-cors',
  headers: myHeaders,
  body: raw,
  // redirect: 'follow'
};

fetch("http://127.0.0.1:5000/synthesize", requestOptions)
  .then(response => response.text())
  .then(result =>{
    console.log(result)
    this.setState(state=>{state.apiresponse=result})

  } )
  .catch(error => console.log('error', error));
};






  render(){
    return (
      <div>
        <div class="row">
      <div class="col-xs-6"></div>
      <div class="container bcontent">
      {/* <img className="logo" src="./public/logo.png" /> */}
     <h1>HearOut</h1> 
     
      <hr />
      <div class="row row-grid align-items-center" id="area" rows="60" column="80"/>
      <div class="card">
      
          <div class="card-header">Generate Text to Speech</div>
          <div class="card-body">
          <div>
      
          {/* <div class="dropdown" >

                              <button class="btn btn-outline-secondary" type="button" data-toggle="dropdown">Language
                              <span class="caret"></span></button>
                              <ul class="dropdown-menu">
                                 <li><a href="#">English</a></li>
                                 <li><a href="#">French</a></li>
                              </ul>
                           </div> */}
 
              <textarea id={'speechtextid'} class="story" name="story" placeholder="Type Here (Maximum 500 Characters)" rows="9" cols="78" maxLength="500">
              </textarea>
              <div><h1>
                {this.state.apiresponse.speech}
              </h1></div>
              
          {/* <div class="card-footer text-muted">500 characters</div> */}
          <div class="container">
  <button type="button" onClick={this.listen} class="btn btn-default" style={{background:"#40E0D0"}}>Listen!</button>
  <button type="button"  class="btn btn-default" style={{background:"#F08080"}}>Reset</button>
</div>
          </div>
      </div>
  </div>
  
</div>



<div class="container-fluid " style={{background:"#DCDCDC"}}>
    <h3>About Us</h3>
    <img src="logo.png" class="rounded-circle" alt="Logo" width="80" height="100"/>
    <h5>Text To Speech delivers human-like, personalized, and engaging user experiencem You now have the ability to translate text into a voice in any language through our Free Online Text to Speech tool. Below you will find some of the languages ​​currently supported on our TTS converter app: English and French</h5>
  </div>
</div>
</div>
       

        
      
    )
  }
}


