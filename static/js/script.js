document.addEventListener("DOMContentLoaded", function () {
    const sidebarItems = document.querySelectorAll(".nav-item");
    const contentSections = document.querySelectorAll(".content-section");
    let charts = {};

    // Sidebar navigation
    sidebarItems.forEach(item => {
        item.addEventListener("click", () => {
            sidebarItems.forEach(i => i.classList.remove("active"));
            contentSections.forEach(sec => sec.style.display = "none");
            item.classList.add("active");
            const targetSection = document.getElementById(item.getAttribute("data-target"));
            if (targetSection) targetSection.style.display = "block";
        });
    });

    function createGradient(ctx) {
        let gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, "rgba(57, 255, 186, 1)");
        gradient.addColorStop(1, "rgba(77, 110, 98, 0.1)");
        return gradient;
    }

    function createOrUpdateChart(chartId, type, labels = [], data = [], colors = [], datasetLabel = "") {
        const ctx = document.getElementById(chartId).getContext("2d");

        if (charts[chartId]) {
            charts[chartId].data.labels = labels;
            charts[chartId].data.datasets[0].data = data;
            charts[chartId].data.datasets[0].label = datasetLabel;
            charts[chartId].update();
        } else {
            charts[chartId] = new Chart(ctx, {
                type: type,
                data: {
                    labels: labels,
                    datasets: [{
                        label: datasetLabel || "Data",
                        data: data,
                        backgroundColor: colors.length ? colors : createGradient(ctx),
                        borderColor: "#fff",
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: "#fff" } }
                    },
                    scales: {
                        x: { ticks: { color: "#fff" }, grid: { color: "rgba(255, 255, 255, 0.1)" } },
                        y: { ticks: { color: "#fff" }, grid: { color: "rgba(255, 255, 255, 0.1)" } }
                    }
                }
            });
        }
    }

    function updateCharts(data) {
        console.log("🔹 Updated Dashboard Data:", data);
    
        const vehicleCount = Array.isArray(data.vehicle_count) ? data.vehicle_count : [];
        const vehicleCategory = Array.isArray(data.vehicle_category) ? data.vehicle_category : [];
        const trafficDensity = Array.isArray(data.traffic_density) ? data.traffic_density : [];
        const highSpeedViolations = Array.isArray(data.high_speed_violations) ? data.high_speed_violations : [];
        const speedDistribution = Array.isArray(data.speed_distribution) ? data.speed_distribution : [];
    
        // Vehicle Count (Line Chart)
        createOrUpdateChart("vehicleCountChart", "line",
            vehicleCount.map(entry => entry.time),
            vehicleCount.map(entry => entry.count),
            ["#39ff14"],
            "Vehicles per 10 second"
        );
    
        // Vehicle Category Distribution (Doughnut)
        const latestCategory = vehicleCategory[vehicleCategory.length - 1];
        if (latestCategory && latestCategory.categories) {
            createOrUpdateChart("vehicleCategoryChart", "doughnut",
                Object.keys(latestCategory.categories),
                Object.values(latestCategory.categories),
                [],
                "Vehicle Types"
            );
        }
    
        // Traffic Density (Bar)
        createOrUpdateChart("trafficDensityChart", "bar",
            vehicleCount.map(entry => entry.time),
            trafficDensity.map(arr => Array.isArray(arr) ? arr.length : 0),
            [],
            "Density Level"
        );
    
        // High-Speed Violations (Bar)
        createOrUpdateChart("highSpeedChart", "bar",
            vehicleCount.map(entry => entry.time),
            highSpeedViolations,
            [],
            "Violations per 10s"
        );
    
        // Speed Distribution (Bar)
        const speeds = speedDistribution.map(entry => entry.average_speed ?? 0);
        const speedBins = ["0-20", "20-40", "40-60", "60-80", "80+"];
        const speedCounts = [0, 0, 0, 0, 0];
    
        speeds.forEach(speed => {
            if (speed <= 20) speedCounts[0]++;
            else if (speed <= 40) speedCounts[1]++;
            else if (speed <= 60) speedCounts[2]++;
            else if (speed <= 80) speedCounts[3]++;
            else speedCounts[4]++;
        });
    
        createOrUpdateChart("speedDistributionChart", "bar",
            speedBins,
            speedCounts,
            [],
            "Speed Bin Count"
        );
    }

    // Load initial data from the template
    const initialData = JSON.parse(document.getElementById("initial-data").textContent);
    updateCharts(initialData);

    // WebSocket for real-time updates
    const socket = io.connect("http://localhost:5000");

    socket.on("connect", () => {
        console.log("✅ WebSocket Connected!");
    });

    socket.on("dashboard_update", (data) => {
        console.log("🟡 WebSocket Data Received:", data);
        updateCharts(data);
    });
});
