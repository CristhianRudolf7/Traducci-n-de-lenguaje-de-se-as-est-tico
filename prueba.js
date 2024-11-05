const ort = require('onnxruntime-web'); // o onnxruntime-node para Node.js

async function loadModel() {
    const session = await ort.InferenceSession.create('./extra_trees_model.onnx');
    return session;
}

const inputData = [150,167,152,157,109,126,173,176,177,173,176,176,172,176,176,72.7308];

// Convertir la lista en un tensor
const tensorInput = new ort.Tensor('float32', new Float32Array(inputData), [1, inputData.length]);

async function runInference(session, inputData) {
    // Ejecutar el modelo con el tensor de entrada
    const feeds = { float_input: tensorInput };  // Reemplaza "float_input" con el nombre real de la entrada del modelo
    const results = await session.run(feeds);
    
    // Asumiendo que la salida es un solo tensor con la predicción
    const output = results.output_name; // Cambia "output_name" al nombre real de la salida
    const predictionIndex = output.data.indexOf(Math.max(...output.data)); // Encuentra el índice con mayor probabilidad

    // Mapea el índice de salida a tu etiqueta correspondiente
    const labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', ' '];
    const predictedLabel = labels[predictionIndex];

    return predictedLabel;
}

async function main() {
    const session = await loadModel();
    const prediction = await runInference(session, inputData);
    console.log("Predicción:", prediction);
}

main();
