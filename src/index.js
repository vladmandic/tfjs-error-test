import * as tf from '@tensorflow/tfjs/dist/tf.es2017.js';

let model;

async function log(msg) {
  const div = document.getElementById('log') || document.createElement('div');
  div.innerHTML += msg + '<br>';
}

async function detect(elem) {
  const imgT = tf.browser.fromPixels(elem);
  log(`Image loaded: "${elem.id}" ${imgT.shape[1]} x ${imgT.shape[0]}`);
  const expandT = tf.expandDims(imgT, 0)
  imgT.dispose();
  const castT = tf.cast(expandT, 'float32');
  expandT.dispose(); 
  const map = ["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0", "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0"]; // boxes, scores
  try {
    const res = await model.executeAsync(castT, map);
    log('OK');
    if (res) {
      log(`Result[0] shape: ${res[0].shape}`)
      log(`Result[1] shape: ${res[1].shape}`)
    }
  } catch (err) {
    console.error(err);
    // log(`Error: ${err.name}: ${err.message}`);
    log('Error');
    log(`<pre>${err.stack}</pre>`);
  }
  castT.dispose();
}

async function main() {
  tf.setBackend('webgl');
  log(`Init TFJS: ${tf.version['tfjs-core']} Backend: ${tf.getBackend()}`);
  await tf.ready();
  log('Loading model...')
  model = await tf.loadGraphModel('./model/model.json');
  log(`Model loaded: ${model.modelUrl}`);
  log(`Engine state: ${tf.engine().state.numBytes} bytes`);
  log('');
  await detect(document.getElementById('imgok'))
  log('');
  await detect(document.getElementById('imgerr'))
  model.dispose();
}

window.onload = main;
