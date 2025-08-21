// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Navigation
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    // Navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            showSection(targetId);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Close mobile menu
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('dateInput').value = today;

    // Prediction button
    document.getElementById('predictBtn').addEventListener('click', getPredictions);
});

// Show specific section
function showSection(sectionId) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
    }
}

// Scroll to section
function scrollToSection(sectionId) {
    showSection(sectionId);
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector(`[href="#${sectionId}"]`).classList.add('active');
}

// Get predictions for selected date
async function getPredictions() {
    const dateInput = document.getElementById('dateInput');
    const selectedDate = dateInput.value;
    
    if (!selectedDate) {
        alert('Please select a date');
        return;
    }

    // Show loading state
    showLoading(true);
    hideResults();

    try {
        // Determine if it's today's date
        const today = new Date().toISOString().split('T')[0];
        const isToday = selectedDate === today;
        
        let response;
        if (isToday) {
            // Use /predict/latest for today
            response = await fetch(`${API_BASE_URL}/model/predict/latest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
        } else {
            // Use /model/date for historical dates
            response = await fetch(`${API_BASE_URL}/model/date?target_date=${selectedDate}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data, isToday);
        
    } catch (error) {
        console.error('Error fetching predictions:', error);
        alert('Error fetching predictions. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Show/hide loading state
function showLoading(show) {
    const loadingState = document.getElementById('loadingState');
    if (show) {
        loadingState.classList.remove('hidden');
    } else {
        loadingState.classList.add('hidden');
    }
}

// Hide results
function hideResults() {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.classList.add('hidden');
}

// Display results
function displayResults(data, isToday) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Update summary
    if (isToday) {
        // For today's predictions
        document.getElementById('targetDate').textContent = new Date().toLocaleDateString();
        document.getElementById('trainingPeriod').textContent = '3 years ending yesterday';
        document.getElementById('modelAccuracy').textContent = 'N/A (Auto-trained)';
    } else {
        // For historical predictions
        document.getElementById('targetDate').textContent = data.target_date;
        document.getElementById('trainingPeriod').textContent = data.training_period;
        document.getElementById('modelAccuracy').textContent = `${(data.training_accuracy * 100).toFixed(2)}%`;
    }

    // Display predictions grid
    displayPredictionsGrid(data.predictions);

    // Create charts
    createPredictionChart(data.predictions);
    createConfidenceChart(data.predictions);

    // Show actual data if available
    if (!isToday && data.actual_data && data.actual_data.data_available) {
        displayActualData(data.actual_data);
    } else {
        hideActualData();
    }

    // Show results
    resultsContainer.classList.remove('hidden');
}

// Display predictions grid
function displayPredictionsGrid(predictions) {
    const gridContainer = document.getElementById('predictionsGrid');
    gridContainer.innerHTML = '';

    const timeSlots = ['08:00', '10:00', '12:00', '14:00', '16:00'];
    
    timeSlots.forEach(time => {
        const prediction = predictions[time];
        if (prediction) {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            
            const movementClass = prediction.movement === 'UP' ? 'movement-up' : 'movement-down';
            const movementIcon = prediction.movement === 'UP' ? '↗' : '↘';
            
            card.innerHTML = `
                <div class="prediction-time">${time}</div>
                <div class="prediction-movement ${movementClass}">${movementIcon} ${prediction.movement}</div>
                <div class="prediction-probability">Probability: ${(prediction.probability * 100).toFixed(1)}%</div>
                <div class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
            `;
            
            gridContainer.appendChild(card);
        }
    });
}

// Create prediction chart
function createPredictionChart(predictions) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.predictionChart) {
        window.predictionChart.destroy();
    }

    const timeSlots = ['08:00', '10:00', '12:00', '14:00', '16:00'];
    const movements = timeSlots.map(time => {
        const pred = predictions[time];
        return pred ? (pred.movement === 'UP' ? 1 : 0) : 0;
    });

    window.predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeSlots,
            datasets: [{
                label: 'Predicted Movement',
                data: movements,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 1,
                        color: '#ffffff',
                        callback: function(value) {
                            return value === 1 ? 'UP' : 'DOWN';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

// Create confidence chart
function createConfidenceChart(predictions) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.confidenceChart) {
        window.confidenceChart.destroy();
    }

    const timeSlots = ['08:00', '10:00', '12:00', '14:00', '16:00'];
    const confidences = timeSlots.map(time => {
        const pred = predictions[time];
        return pred ? pred.confidence * 100 : 0;
    });

    window.confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: timeSlots,
            datasets: [{
                label: 'Confidence Level (%)',
                data: confidences,
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: '#00d4ff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

// Display actual data table
function displayActualData(actualData) {
    const actualDataSection = document.getElementById('actualDataSection');
    const tableBody = document.getElementById('actualDataTable').getElementsByTagName('tbody')[0];
    
    tableBody.innerHTML = '';

    // Add target data if available
    if (actualData.target_data) {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td><strong>${actualData.target_data.date}</strong></td>
            <td>${actualData.target_data.open.toFixed(2)}</td>
            <td>${actualData.target_data.high.toFixed(2)}</td>
            <td>${actualData.target_data.low.toFixed(2)}</td>
            <td>${actualData.target_data.close.toFixed(2)}</td>
            <td>${actualData.target_data.volume.toLocaleString()}</td>
            <td>${(actualData.target_data.returns * 100).toFixed(2)}%</td>
        `;
        row.style.backgroundColor = 'rgba(0, 212, 255, 0.1)';
    }

    // Add context data
    actualData.context_data.forEach(dataPoint => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${dataPoint.date}</td>
            <td>${dataPoint.open.toFixed(2)}</td>
            <td>${dataPoint.high.toFixed(2)}</td>
            <td>${dataPoint.low.toFixed(2)}</td>
            <td>${dataPoint.close.toFixed(2)}</td>
            <td>${dataPoint.volume.toLocaleString()}</td>
            <td>${(dataPoint.returns * 100).toFixed(2)}%</td>
        `;
    });

    actualDataSection.classList.remove('hidden');
}

// Hide actual data section
function hideActualData() {
    const actualDataSection = document.getElementById('actualDataSection');
    actualDataSection.classList.add('hidden');
}
