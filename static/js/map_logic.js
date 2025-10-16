// File Path: Disaster_ML_Fusion/static/js/map_logic.js

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize the Map
    // Start centered roughly on the global disaster area, zoomed out (Zoom 2)
    const map = L.map('map').setView([30, 0], 2); 

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    let riskMarker = null; // To hold the ML prediction marker

    // --- Add Historical Disaster Markers ---
    historicalData.forEach(event => {
        // Only plot events with valid coordinates
        if (event.Latitude && event.Longitude) {
            const isHighSeverity = event.Severity_Level === 1;
            const markerColor = isHighSeverity ? 'red' : 'orange';
            
            // Custom Icon for Historical Events
            const eventIcon = L.divIcon({
                className: `history-icon ${markerColor}`,
                html: '⚫', 
                iconSize: [8, 8],
                iconAnchor: [4, 4] // Center the icon
            });

            L.marker([event.Latitude, event.Longitude], {icon: eventIcon}).addTo(map)
                .bindPopup(`
                    <b>${event.Disaster_Type} (${event.Start_Date})</b><br>
                    Country: ${event.Country}<br>
                    Severity: ${isHighSeverity ? 'High' : 'Low'}<br>
                    Total Affected: ${event.Total_Affected}
                `);
        }
    });

    // 2. Handle the Risk Prediction Form Submission (API Call)
    document.getElementById('risk-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const selectedArea = document.getElementById('area-select').value;
        
        if (selectedArea === 'Select') {
            alert("Please select an area for prediction.");
            return;
        }

        // Mock coordinates for the selected area (used to move the map)
        const coords = selectedArea === 'Downtown' ? [35.68, 139.75] : [40.71, -74.00];

        // Call the Flask API endpoint
        const response = await fetch('/api/predict_risk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                area: selectedArea,
                latitude: coords[0],
                longitude: coords[1],
            })
        });

        const data = await response.json();
        
        // 3. Update the Sidebar with Prediction Results
        document.getElementById('risk-level').textContent = data.risk_level;
        document.getElementById('risk-probability').textContent = `${(data.probability * 100).toFixed(1)}%`;
        document.getElementById('risk-message').textContent = data.message;
        
        // Change the color of the risk level text
        document.getElementById('risk-level').style.color = data.risk_level === 'HIGH' ? '#e74c3c' : '#2ecc71';
        
        // 4. Update the Map with the Predicted Risk Area
        const markerColor = data.risk_level === 'HIGH' ? 'red' : 'green';
        const markerIcon = L.divIcon({
            className: `risk-icon ${markerColor}`,
            html: '🚨', 
            iconSize: [30, 30]
        });

        if (riskMarker) {
            map.removeLayer(riskMarker);
        }

        riskMarker = L.marker(coords, {icon: markerIcon}).addTo(map)
            .bindPopup(`<b>ML Prediction:</b> ${data.message}`)
            .openPopup();
            
        map.setView(coords, 8); // Zoom to the predicted area
    });
});