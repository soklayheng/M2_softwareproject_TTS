import React, {Component } from "react";
import {
  Link 
} from 'react-router-dom' 

export default class Header extends Component{

  render(){
    return (
      <div className="headerarea">
        {/* <img className="logo" src="https://pngimg.com/uploads/bmw_logo/bmw_logo_PNG19705.png" /> */}
        {/* <img src="logo.png" alt="Logo" width="80" height="100"/> */}
        {/* <ul className ="menu-ul">
          <li><Link to="/">Home </Link></li>
          <li><Link to="/about">About </Link></li>
          <li><Link to="/contact">Contact </Link></li>
        </ul> */}
      </div>
    
    )
  }
}