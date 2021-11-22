import React, {Component } from "react";
// function fetch() {
//   return new Promise(resolve => setTimeout(() => resolve(42), 1000));
// }
export async function fetchVoice()
{ 
  console.log("calling api");
  // return fetch("http://localhost:5000/synthesize");
  //return fetch("https://jsonplaceholder.typicode.com/todos/1");

}
export default class Home extends Component {

listen=()=>{
  fetch("https://jsonplaceholder.typicode.com/todos/1", {
    method: "GET",
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
  }
)
.then((res) => console.log(res))
.catch(error=>console.log(error));
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
 
              <textarea id="story" class="story" name="story" placeholder="Type Here (Maximum 500 Characters)"rows="9" cols="78" maxLength="500"></textarea>
              
          {/* <div class="card-footer text-muted">500 characters</div> */}
          <div class="container">
  <button type="button" onClick={this.listen} class="btn btn-default" style={{background:"#40E0D0"}}>Listen!</button>
  <button type="button" class="btn btn-default" style={{background:"#F08080"}}>Reset</button>
</div>
          </div>
      </div>
  </div>
  
</div>



<div class="container-fluid " style={{background:"#DCDCDC"}}>
    <h3>About Us</h3>
    <img src="logo.png" class="rounded-circle" alt="Logo" width="80" height="100"/>
    <h5>Text To Speech delivers human-like, personalized, and engaging user experiencem You now have the ability to translate text into a voice in any language through our Free Online Text to Speech tool. Below you will find some of the languages ​​currently supported on our TTS converter app</h5>
  </div>
</div>
</div>
       

        
      
    )
  }
}


