import * as tf from '@tensorflow/tfjs';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultText = document.getElementById('result');

const classNames = [
  'Apple Braeburn',
  'Apple Granny Smith',
  'Apricot',
  'Avocado',
  'Banana',
  'Blueberry',
  'Cactus fruit',
  'Cantaloupe',
  'Cherry',
  'Clementine',
  'Corn',
  'Cucumber Ripe',
  'Grape Blue',
  'Kiwi',
  'Lemon',
  'Limes',
  'Mango',
  'Onion White',
  'Orange',
  'Papaya',
  'Passion Fruit',
  'Peach',
  'Pear',
  'Pepper Green',
  'Pepper Red',
  'Pineapple',
  'Plum',
  'Pomegranate',
  'Potato Red',
  'Raspberry',
  'Strawberry',
  'Tomato',
  'Watermelon'
];

let model = null;

async function loadModel() {
  try {
    model = await tf.loadLayersModel('/model/model.json');
    resultText.textContent = 'Modelo cargado. Cámara lista.';
    console.log('Modelo cargado correctamente');
  } catch (error) {
    console.error('Error cargando el modelo:', error);
    resultText.textContent = 'No se pudo cargar el modelo. Verifica /model/model.json';
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
  } catch (error) {
    console.error('Error al acceder a la cámara:', error);
    resultText.textContent = 'Necesitas permitir el acceso a la cámara.';
  }
}

function predictFrame() {
  if (!model || video.readyState < 2) {
    requestAnimationFrame(predictFrame);
    return;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = tf.browser.fromPixels(canvas);
  const resized = tf.image.resizeBilinear(image, [160, 160]);
  const normalized = resized.toFloat().div(255.0);
  const batch = normalized.expandDims(0);

  const prediction = model.predict(batch);
  const probabilities = prediction.dataSync();
  const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[predictedIndex] * 100;
  const label = classNames[predictedIndex] ?? `Clase ${predictedIndex}`;

  resultText.textContent = `${label} (${confidence.toFixed(1)}%)`;

  image.dispose();
  resized.dispose();
  normalized.dispose();
  batch.dispose();
  prediction.dispose();

  requestAnimationFrame(predictFrame);
}

async function init() {
  await loadModel();
  await startCamera();

  if (model) {
    requestAnimationFrame(predictFrame);
  }
}

init();
