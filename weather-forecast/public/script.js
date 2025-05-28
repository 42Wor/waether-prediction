class MyLinearRegression {
    constructor() {
        this.coef_ = null;
        this.intercept_ = null;
    }

    load(modelData) {
        this.coef_ = modelData.coef;
        this.intercept_ = modelData.intercept;
        return this;
    }

    predict(input) {
        if (!Array.isArray(input) || input.length === 0) {
            throw new Error('Input must be a non-empty array');
        }

        // If input is 2D array, use the first row
        const features = Array.isArray(input[0]) ? input[0] : input;

        if (features.length !== this.coef_.length) {
            throw new Error(`Input feature count (${features.length}) does not match model (${this.coef_.length})`);
        }

        let prediction = this.intercept_;
        for (let i = 0; i < this.coef_.length; i++) {
            prediction += this.coef_[i] * features[i];
        }

        return prediction;
    }
}

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Check for saved theme preference or use dark theme as default
const currentTheme = localStorage.getItem('theme') || 'dark';
body.classList.add(`${currentTheme}-theme`);

// Update the toggle button based on current theme
updateThemeToggle(currentTheme);

themeToggle.addEventListener('click', () => {
    if (body.classList.contains('dark-theme')) {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        localStorage.setItem('theme', 'light');
        updateThemeToggle('light');
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
        updateThemeToggle('dark');
    }

    // Update charts to match theme
    updateChartThemes();
});

function updateThemeToggle(theme) {
    const icon = themeToggle.querySelector('i');
    const text = themeToggle.querySelector('span');

    if (theme === 'dark') {
        icon.className = 'fas fa-moon';
        text.textContent = 'Dark Mode';
    } else {
        icon.className = 'fas fa-sun';
        text.textContent = 'Light Mode';
    }
}

// Store chart references for theme updates
const charts = {};

// DOM elements
const weatherCardsContainer = document.querySelector('.weather-cards');
const days = [1, 2, 3, 4, 5, 6, 7];

// Weather parameters
const weatherParams = [{id: 'max-temp', name: 'Max Temp', unit: '°C', color: '#e74c3c'}, {
    id: 'min-temp',
    name: 'Min Temp',
    unit: '°C',
    color: '#3498db'
}, {id: 'humidity', name: 'Humidity', unit: '%', color: '#2ecc71'}, {
    id: 'pressure',
    name: 'Pressure',
    unit: 'hPa',
    color: '#9b59b6'
}, {id: 'precip', name: 'Precipitation', unit: 'mm', color: '#1abc9c'}, {
    id: 'wind',
    name: 'Wind Speed',
    unit: 'km/h',
    color: '#f39c12'
}];

// Create weather cards for days 2-7
for (let i = 1; i < 7; i++) {
    const card = document.createElement('div');
    card.className = 'weather-card';
    card.id = `day${i + 1}`;

    const html = `
        <h2>Day ${i + 1}</h2>
        <div class="weather-data">
          ${weatherParams.map(param => `
            <div class="weather-item">
              <span class="label">${param.name}:</span>
              <span class="value" id="${param.id}-${i + 1}">--</span>
            </div>
          `).join('')}
        </div>
      `;

    card.innerHTML = html;
    weatherCardsContainer.appendChild(card);
}

// Load models and make predictions
async function loadModelsAndPredict() {
    try {
        // Load all models
        const models = {
            maxTemp: new MyLinearRegression(),
            minTemp: new MyLinearRegression(),
            humidity: new MyLinearRegression(),
            precipitation: new MyLinearRegression(),
            pressure: new MyLinearRegression(),
            wind: new MyLinearRegression()
        };

        // Fetch model data
        const modelData = await Promise.all([fetch('/json/Max_Temp.json').then(res => res.json()), fetch('/json/Min_Temp.json').then(res => res.json()), fetch('/json/Humidity.json').then(res => res.json()), fetch('/json/Precipitation.json').then(res => res.json()), fetch('/json/Pressure.json').then(res => res.json()), fetch('/json/Wind Speed.json').then(res => res.json())]);

        // Initialize models with fetched data
        models.maxTemp.load(modelData[0]);
        models.minTemp.load(modelData[1]);
        models.humidity.load(modelData[2]);
        models.precipitation.load(modelData[3]);
        models.pressure.load(modelData[4]);
        models.wind.load(modelData[5]);

        // Initial weather data [Max_Temp, Min_Temp, Humidity, Pressure, Precipitation, Wind]
        let current = [38.8, 28.5, 24.0, 1014.8, 0.0, 14.6];
        const predictions = [];

        // Predict for 7 days
        for (let day = 1; day <= 7; day++) {
            const nextDay = [0, 0, 0, 0, 0, 0]; // Initialize next day's features

            // Predict each feature
            // Max_Temp (index0): exclude current[0]
            input_max = [current[1], current[2], current[3], current[4], current[5]];
            nextDay[0] = models.maxTemp.predict(input_max);

            // Min_Temp (index1): exclude current[1]
            input_min = [current[0], current[2], current[3], current[4], current[5]];
            nextDay[1] = models.minTemp.predict(input_min);

            // Humidity (index2): exclude current[2]
            input_hum = [current[0], current[1], current[3], current[4], current[5]];
            nextDay[2] = models.humidity.predict(input_hum);

            // Pressure (index3): exclude current[3]
            input_pres = [current[0], current[1], current[2], current[4], current[5]];
            nextDay[3] = models.pressure.predict(input_pres);

            // Precipitation (index4): exclude current[4], clamp negative values
            input_precip = [current[0], current[1], current[2], current[3], current[5]];
            nextDay[4] = Math.max(0.0, models.precipitation.predict(input_precip));

            // Wind (index5): exclude current[5]
            input_wind = [current[0], current[1], current[2], current[3], current[4]];
            nextDay[5] = models.wind.predict(input_wind);

            // Update UI for this day
            updateWeatherCard(day, nextDay);

            // Store predictions for charts
            predictions.push([...nextDay]);

            // Use predictions for the next day
            current = nextDay;
        }

        // Create charts
        createCharts(predictions);
    } catch (error) {
        console.error('Error loading models or making predictions:', error);
        alert('Failed to load weather data. Please try again later.');
    }
}

function updateWeatherCard(day, data) {
    weatherParams.forEach((param, index) => {
        const element = document.getElementById(`${param.id}-${day}`);
        if (element) {
            element.textContent = `${data[index].toFixed(1)}${param.unit}`;
        }
    });
}

function createCharts(predictions) {
    const days = [1, 2, 3, 4, 5, 6, 7];

    // Extract data for each parameter
    const maxTemps = predictions.map(day => day[0]);
    const minTemps = predictions.map(day => day[1]);
    const humidities = predictions.map(day => day[2]);
    const pressures = predictions.map(day => day[3]);
    const precipitations = predictions.map(day => day[4]);
    const winds = predictions.map(day => day[5]);

    // Get current theme to set appropriate chart colors
    const isDarkTheme = body.classList.contains('dark-theme');
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDarkTheme ? '#ecf0f1' : '#333';

    // Common chart options
    const commonOptions = {
        responsive: true, maintainAspectRatio: true, plugins: {
            legend: {
                labels: {
                    color: textColor
                }
            }, title: {
                display: true, color: textColor, font: {size: 16}
            }
        }, scales: {
            x: {
                grid: {
                    color: gridColor
                }, ticks: {
                    color: textColor
                }
            }, y: {
                grid: {
                    color: gridColor
                }, ticks: {
                    color: textColor
                }
            }
        }
    };

    // Temperature Chart
    charts.temperature = new Chart(document.getElementById('temperature-chart'), {
        type: 'line', data: {
            labels: days, datasets: [{
                label: 'Max Temperature (°C)',
                data: maxTemps,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.3,
                fill: true
            }, {
                label: 'Min Temperature (°C)',
                data: minTemps,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.3,
                fill: true
            }]
        }, options: {
            ...commonOptions, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Temperature Forecast'
                }
            }, scales: {
                ...commonOptions.scales, y: {
                    ...commonOptions.scales.y, title: {
                        display: true, text: 'Temperature (°C)', color: textColor
                    }
                }
            }
        }
    });

    // Humidity Chart
    charts.humidity = new Chart(document.getElementById('humidity-chart'), {
        type: 'line', data: {
            labels: days, datasets: [{
                label: 'Humidity (%)',
                data: humidities,
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                tension: 0.3,
                fill: true
            }]
        }, options: {
            ...commonOptions, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Humidity Forecast'
                }
            }, scales: {
                ...commonOptions.scales, y: {
                    ...commonOptions.scales.y, title: {
                        display: true, text: 'Humidity (%)', color: textColor
                    }
                }
            }
        }
    });

    // Pressure Chart
    charts.pressure = new Chart(document.getElementById('pressure-chart'), {
        type: 'line', data: {
            labels: days, datasets: [{
                label: 'Pressure (hPa)',
                data: pressures,
                borderColor: '#9b59b6',
                backgroundColor: 'rgba(155, 89, 182, 0.1)',
                tension: 0.3,
                fill: true
            }]
        }, options: {
            ...commonOptions, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Pressure Forecast'
                }
            }, scales: {
                ...commonOptions.scales, y: {
                    ...commonOptions.scales.y, title: {
                        display: true, text: 'Pressure (hPa)', color: textColor
                    }
                }
            }
        }
    });

    // Precipitation Chart
    charts.precipitation = new Chart(document.getElementById('precipitation-chart'), {
        type: 'bar', data: {
            labels: days, datasets: [{
                label: 'Precipitation (mm)',
                data: precipitations,
                backgroundColor: '#1abc9c',
                borderColor: '#16a085',
                borderWidth: 1
            }]
        }, options: {
            ...commonOptions, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Precipitation Forecast'
                }
            }, scales: {
                ...commonOptions.scales, y: {
                    ...commonOptions.scales.y, beginAtZero: true, title: {
                        display: true, text: 'Precipitation (mm)', color: textColor
                    }
                }
            }
        }
    });

    // Wind Chart
    charts.wind = new Chart(document.getElementById('wind-chart'), {
        type: 'line', data: {
            labels: days, datasets: [{
                label: 'Wind Speed (km/h)',
                data: winds,
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.3,
                fill: true
            }]
        }, options: {
            ...commonOptions, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Wind Speed Forecast'
                }
            }, scales: {
                ...commonOptions.scales, y: {
                    ...commonOptions.scales.y, title: {
                        display: true, text: 'Wind Speed (km/h)', color: textColor
                    }
                }
            }
        }
    });

    // Combined Chart
    charts.combined = new Chart(document.getElementById('combined-chart'), {
        type: 'line', data: {
            labels: days, datasets: [{
                label: 'Max Temp (°C)',
                data: maxTemps,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.3,
                yAxisID: 'y'
            }, {
                label: 'Min Temp (°C)',
                data: minTemps,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.3,
                yAxisID: 'y'
            }, {
                label: 'Humidity (%)',
                data: humidities,
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                tension: 0.3,
                yAxisID: 'y1'
            }, {
                label: 'Wind (km/h)',
                data: winds,
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.3,
                yAxisID: 'y2'
            }]
        }, options: {
            ...commonOptions, interaction: {
                mode: 'index', intersect: false
            }, plugins: {
                ...commonOptions.plugins, title: {
                    ...commonOptions.plugins.title, text: 'Combined Forecast'
                }
            }, scales: {
                x: {
                    ...commonOptions.scales.x
                }, y: {
                    type: 'linear', display: true, position: 'left', title: {
                        display: true, text: 'Temperature (°C)', color: textColor
                    }, grid: {
                        color: gridColor
                    }, ticks: {
                        color: textColor
                    }
                }, y1: {
                    type: 'linear', display: true, position: 'right', title: {
                        display: true, text: 'Humidity (%)', color: textColor
                    }, grid: {
                        drawOnChartArea: false, color: gridColor
                    }, ticks: {
                        color: textColor
                    }
                }, y2: {
                    type: 'linear', display: true, position: 'right', title: {
                        display: true, text: 'Wind (km/h)', color: textColor
                    }, grid: {
                        drawOnChartArea: false, color: gridColor
                    }, ticks: {
                        color: textColor
                    }, afterFit: function (axis) {
                        axis.paddingRight = 50;
                    }
                }
            }
        }
    });
}

function updateChartThemes() {
    const isDarkTheme = body.classList.contains('dark-theme');
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDarkTheme ? '#ecf0f1' : '#333';

    Object.values(charts).forEach(chart => {
        chart.options.scales.x.grid.color = gridColor;
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.y.grid.color = gridColor;
        chart.options.scales.y.ticks.color = textColor;

        if (chart.options.scales.y1) {
            chart.options.scales.y1.grid.color = gridColor;
            chart.options.scales.y1.ticks.color = textColor;
            chart.options.scales.y1.title.color = textColor;
        }

        if (chart.options.scales.y2) {
            chart.options.scales.y2.grid.color = gridColor;
            chart.options.scales.y2.ticks.color = textColor;
            chart.options.scales.y2.title.color = textColor;
        }

        chart.options.plugins.title.color = textColor;
        chart.options.plugins.legend.labels.color = textColor;

        chart.update();
    });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', loadModelsAndPredict);
