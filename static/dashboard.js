function getQueryParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

function updateQueryParam(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.replaceState({}, '', url); // no page reload
}

function renderChart({ container, datasets, stacked = false, title, formatX = null, formatY = null }) {
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 400;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');

    const allX = datasets.flatMap(dataset => dataset.data.map(point => point.x));
    const allY = datasets.flatMap(dataset => dataset.data.map(point => point.y));

    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    let minY = Math.min(...allY);
    let maxY = Math.max(...allY);

    // Add padding to Y axis
    const yRange = maxY - minY;
    const padding = yRange === 0 ? 0.1 : yRange * 0.05; // handle flat lines

    minY -= padding;
    maxY += padding;

    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            stacked: stacked,
            parsing: false,
            plugins: {
                title: title
                    ? {
                        display: true,
                        text: title,
                        font: { size: 18, weight: 'bold' },
                        padding: { top: 10, bottom: 20 }
                    }
                    : false,
                tooltip: {
                    callbacks: {
                        label: context => {
                            const label = context.dataset.label || '';
                            const x = context.raw.x;
                            const y = context.raw.y;
                            return `${label}: (${formatX ? formatX(x) : x}, ${formatY ? formatY(y) : y})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: minX,
                    max: maxX,
                    ticks: {
                        count: 10,
                        callback: val => formatX ? formatX(val) : parseFloat(val).toFixed(2)
                    },
                    title: {
                        display: true,
                        text: 'X Axis'
                    }
                },
                y: {
                    type: 'linear',
                    min: minY,
                    max: maxY,
                    stacked: stacked,
                    ticks: {
                        callback: val => formatY ? formatY(val) : val
                    },
                    title: {
                        display: true,
                        text: 'Y Axis'
                    }
                }
            }
        }
    });
}

function renderLineChart({ container, datalists, labelFn, title, formatX = null, formatY = null }) {
    const datasets = datalists.map((datalist, index) => datalist ? {
        label: labelFn(index),
        data: datalist,
        borderWidth: 2,
        fill: false,
        tension: 0.3
    } : null).filter(d => d != null);

    renderChart({container, datasets, labelFn, title, formatX, formatY});
}

function renderHistogramChart({ container, histograms, labelFn, title, formatX = null, formatY = null }) {
    const datasets = histograms.map((hist, index) => hist ? {
        label: labelFn(index),
        data: hist.x.map((xVal, i) => ({
            x: xVal,
            y: hist.y[i]
        })),
        borderWidth: 2,
        fill: true,
        tension: 0.3
    } : null).filter(d => d != null);

    renderChart({container, datasets, stacked: true, title, formatX, formatY});
}

async function refresh() {
    // Take inputs and store
    const modelId = document.getElementById('model-id').value.trim();
    if (!modelId) {
        alert("Please enter a model ID.");
        return;
    }
    updateQueryParam('model_id', modelId);

    const layerFilter = document.getElementById('layer-filter').value.trim().toLowerCase();
    updateQueryParam('layer', layerFilter);

    // Fetch progress
    const progressResponse = await fetch(`/progress?model_id=${encodeURIComponent(modelId)}`);
    if (!progressResponse.ok) {
        alert("Failed to fetch progress. Check model ID.");
        return;
    }

    const progressData = await progressResponse.json();
    if (!progressData) {
        // No progress recorded yet for model
        return;
    }

    // Fetch stats
    const statsResponse = await fetch(`/stats?model_id=${encodeURIComponent(modelId)}`);
    if (!statsResponse.ok) {
        alert("Failed to fetch stats. Check model ID.");
        return;
    }

    const statsData = await statsResponse.json();
    
    // Clear previous data
    const container = document.getElementById('data-container');
    container.innerHTML = '';

    // Render cost progress
    const headerCostProgress = document.createElement('h2');
    headerCostProgress.textContent = `Cost progress for model ${modelId}`;
    container.appendChild(headerCostProgress);
    const headerAvgCost = document.createElement('h3');
    headerAvgCost.textContent = `Average Cost: ${progressData.average_cost}`;
    container.appendChild(headerAvgCost);
    renderLineChart({
        container,
        datalists: [progressData.progress.map(p => ({
            x: p.epoch,
            y: Math.log10(p.cost),
        }))],
        labelFn: () => `Cost progress`,
        title: "Cost Progression (Log Scale)",
        formatX: x => x.toFixed(0),
        formatY: y => y.toFixed(4),
    });
    renderLineChart({
        container,
        datalists: [progressData.average_cost_history.map((avg_cost, i) => ({
            x: i,
            y: Math.log10(avg_cost),
        }))],
        labelFn: () => `Overall Average Cost`,
        title: "Average Cost Overall (Log Scale)",
        formatX: x => x.toFixed(0),
        formatY: y => y.toFixed(4),
    });

    // layer to index lookup and filtering for corresponding weight index
    const layerToIndex = new Map(statsData ? statsData.layers.map((layer, i) => [layer, i]): []);
    const layerFilters = layerFilter.split(',').map(f => f.trim()).filter(f => f.length > 0);
    const layerFilterFunc = (l, i) => (
        layerFilters.length === 0 || layerFilters.some(f => (f == i) || l?.algo.includes(f))
    );
    const layerIndexFilterFunc = i => layerFilterFunc(statsData ? statsData.layers[i] : null, i);

    // Render weight updates
    const headerWeightUpdates = document.createElement('h2');
    headerWeightUpdates.textContent = `Weight updates for model ${modelId}`;
    container.appendChild(headerWeightUpdates);
    const weightUpdateDatalists = [];
    progressData.progress.forEach(p => {
        p.weight_upd_ratio.forEach((wupdr, i) => {
            if (wupdr && layerIndexFilterFunc(i)) {
                (weightUpdateDatalists[i] ??= []).push({
                    x: p.epoch,
                    y: Math.log10(wupdr),
                });
            }
        });
    });
    renderLineChart({
        container,
        datalists: weightUpdateDatalists,
        labelFn: (i) => `Weights ${i}`,
        title: "Weight Update Std Ratio (Log Scale)",
        formatX: x => x.toFixed(0),
        formatY: y => y.toFixed(4),
    });

    if (!statsData) {
        // No stats recorded yet for model
        return;
    }
    
    // Apply optional layer filter
    const layers = statsData.layers.filter(layerFilterFunc);

    // Render activation stats
    const headerActivations = document.createElement('h2');
    headerActivations.textContent = `Activations for model ${modelId}`;
    container.appendChild(headerActivations);
    layers.forEach(layer => {
        const activation = layer.activation;
        const mean = activation.mean.toFixed(2);
        const std = activation.std.toFixed(2);
        const saturated = (activation.saturated * 100).toFixed(1) + '%';
        
        const header = document.createElement('h3');
        header.textContent = `Layer ${layerToIndex.get(layer)} (${layer.algo}): mean ${mean} std ${std} saturated: ${saturated}`;
        container.appendChild(header);
    });
    renderHistogramChart({
        container,
        histograms: layers.map(layer => layer.activation.histogram),
        labelFn: (i) => `Layer ${layerToIndex.get(layers[i])} (${layers[i].algo})`,
        title: 'Activation Distribution',
        formatY: y => y.toFixed(4),
    });

    // Render gradient stats
    const headerGradients = document.createElement('h2');
    headerGradients.textContent = `Gradients for model ${modelId}`;
    container.appendChild(headerGradients);
    layers.forEach(layer => {
        const gradient = layer.gradient;
        if (gradient) {
            const mean = gradient.mean.toExponential(6);
            const std = gradient.std.toExponential(6);
            
            const header = document.createElement('h3');
            header.textContent = `Layer ${layerToIndex.get(layer)} (${layer.algo}): mean ${mean} std ${std}`;
            container.appendChild(header);
        }
    });
    renderHistogramChart({
        container,
        histograms: layers.map(layer => layer.gradient?.histogram),
        labelFn: i => `Layer ${layerToIndex.get(layers[i])} (${layers[i].algo})`,
        title: 'Gradient Distribution',
        formatX: x => x.toFixed(6),
        formatY: y => y.toFixed(4),
    });


    // Render weight stats
    const headerWeights = document.createElement('h2');
    headerWeights.textContent = `Weights for model ${modelId}`;
    container.appendChild(headerWeights);
    const weightStats = statsData.weights.map((w, i) => layerIndexFilterFunc(i) ? w : null);
    weightStats.forEach((w, i) => {
        if (w) {
            const mean = w.gradient.mean.toExponential(6);
            const std = w.gradient.std.toExponential(6);
            const ratio = (std / w.data.std).toExponential(6);
            
            const header = document.createElement('h3');
            header.textContent = `Weights ${i} - ${w.shape}: mean ${mean} std ${std}  grad:data ratio ${ratio}`;
            container.appendChild(header);
        }
    });
    renderHistogramChart({
        container,
        histograms: weightStats.map(w => w?.gradient.histogram),
        labelFn: i => `Weights ${i} - ${weightStats[i].shape}`,
        title: 'Weight Gradient Distribution',
        formatX: x => x.toFixed(3),
        formatY: y => y.toFixed(4),
    });
}

// On page load: auto-fill inputs
window.onload = () => {
    const modelId = getQueryParam('model_id');
    if (modelId) {
        document.getElementById('model-id').value = modelId;
    }
    const layerFilter = getQueryParam('layer');
    if (layerFilter) {
        document.getElementById('layer-filter').value = layerFilter
    }
};
