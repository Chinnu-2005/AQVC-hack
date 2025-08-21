# Quantum ML FTSE 100 Predictor - React Frontend

A modern React.js frontend for the Quantum Machine Learning FTSE 100 Predictor, featuring a beautiful quantum-themed design with interactive charts and real-time predictions.

## Features

- **Quantum Theme Design**: Dark gradient background with cyan accents
- **Responsive Layout**: Works perfectly on desktop and mobile devices
- **Interactive Charts**: Real-time prediction visualization using Chart.js
- **Date Selection**: Choose any date for predictions (today or historical)
- **Real-time API Integration**: Seamless connection with the Python backend
- **Loading States**: Beautiful quantum-themed loading animations
- **Error Handling**: User-friendly error messages

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Navbar.js
│   │   ├── Home.js
│   │   ├── About.js
│   │   ├── Dashboard.js
│   │   ├── LoadingSpinner.js
│   │   ├── PredictionGrid.js
│   │   ├── PredictionCharts.js
│   │   └── ActualDataTable.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

## Installation

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

The application will open at `http://localhost:3000`

## Usage

### Prerequisites
- Make sure the Python backend is running on `http://localhost:8000`
- The backend should be started with: `python main.py` from the `backend` directory

### Features

1. **Home Section**: 
   - Hero section with quantum animations
   - Statistics about the system
   - Call-to-action button to navigate to dashboard

2. **About Section**:
   - Information about quantum technology
   - Feature cards explaining the system capabilities

3. **Dashboard Section**:
   - Date picker for selecting prediction dates
   - Real-time predictions for 5 time intervals (8:00, 10:00, 12:00, 14:00, 16:00)
   - Interactive charts showing movement patterns and confidence levels
   - Historical data comparison for past dates

### API Endpoints Used

- `POST /model/predict/latest` - For today's predictions
- `POST /model/date?target_date=YYYY-MM-DD` - For historical date predictions

## Technologies Used

- **React 18**: Modern React with hooks and functional components
- **Chart.js**: Interactive charts for data visualization
- **CSS3**: Custom styling with quantum theme
- **Fetch API**: For backend communication

## Styling

The application uses a custom quantum theme with:
- Dark gradient backgrounds
- Cyan/blue accent colors (#00d4ff)
- Glassmorphism effects
- Smooth animations and transitions
- Responsive design for all screen sizes

## Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

### Component Structure

- **App.js**: Main application component with routing
- **Navbar.js**: Navigation component with mobile menu
- **Home.js**: Landing page with hero section
- **About.js**: Information about the quantum system
- **Dashboard.js**: Main prediction interface
- **PredictionGrid.js**: Grid of prediction cards
- **PredictionCharts.js**: Chart.js integration for visualizations
- **ActualDataTable.js**: Table for historical data display
- **LoadingSpinner.js**: Quantum-themed loading animation

## Deployment

To build for production:

```bash
npm run build
```

The build files will be created in the `build/` directory, ready for deployment to any static hosting service.

## Troubleshooting

1. **Backend Connection Issues**: Ensure the Python backend is running on port 8000
2. **Chart Display Issues**: Check that Chart.js dependencies are properly installed
3. **Styling Issues**: Verify that all CSS files are properly imported

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Quantum ML FTSE 100 Predictor system.
