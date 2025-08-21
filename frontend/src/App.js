import React, { useState } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import Home from './components/Home';
import About from './components/About';
import Dashboard from './components/Dashboard';

function App() {
  const [activeSection, setActiveSection] = useState('home');

  const handleNavigation = (section) => {
    setActiveSection(section);
  };

  return (
    <div className="App">
      <Navbar activeSection={activeSection} onNavigate={handleNavigation} />
      
      {activeSection === 'home' && <Home onNavigate={handleNavigation} />}
      {activeSection === 'about' && <About />}
      {activeSection === 'dashboard' && <Dashboard />}
    </div>
  );
}

export default App;
