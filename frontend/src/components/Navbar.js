import React, { useState } from 'react';
import './Navbar.css';

const Navbar = ({ activeSection, onNavigate }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleNavClick = (section) => {
    onNavigate(section);
    setIsMobileMenuOpen(false);
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-logo">
          <span className="quantum-icon">⚛</span>
          <span className="logo-text">Quantum ML</span>
        </div>
        
        <ul className={`nav-menu ${isMobileMenuOpen ? 'active' : ''}`}>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSection === 'home' ? 'active' : ''}`}
              onClick={() => handleNavClick('home')}
            >
              Home
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSection === 'about' ? 'active' : ''}`}
              onClick={() => handleNavClick('about')}
            >
              About
            </button>
          </li>
          <li className="nav-item">
            <button 
              className={`nav-link ${activeSection === 'dashboard' ? 'active' : ''}`}
              onClick={() => handleNavClick('dashboard')}
            >
              Dashboard
            </button>
          </li>
        </ul>
        
        <div className="hamburger" onClick={toggleMobileMenu}>
          <span className={`bar ${isMobileMenuOpen ? 'active' : ''}`}></span>
          <span className={`bar ${isMobileMenuOpen ? 'active' : ''}`}></span>
          <span className={`bar ${isMobileMenuOpen ? 'active' : ''}`}></span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
