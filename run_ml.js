function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data)

        const inputs = data.map(d => [d.x, d.y, d.tof])
        const labels = data.map(d => [d.dX, d.dY])

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 3])
        const labelTensor = tf.tensor2d(labels, [labels.length, 2])
        
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return min/max bounds to use later
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    })
}

function createModel() {
    const model = tf.sequential()

    model.add(tf.layers.dense({inputShape: [3], units: 50, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 400, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 200, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 100, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 50, activation: 'relu'}))
    model.add(tf.layers.dense({units: 2}))

    return model
  }
convertToTensor(data)
createModel()
let model = createModel()
let tensorData
async function run(data) {
    tensorData = convertToTensor(data)
    const {inputs, labels} = tensorData
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.absoluteDifference,
        metrics: ['mse']
    })
    const batchSize = 100
    const epochs = 5
    await model.fit(inputs, labels, {
        batchSize,
        epochs,
        callbacks: {onEpochEnd, onBatchEnd}
    })
    console.log('Done Training');
    model.summary()

    
}
function onEpochEnd(batch, logs) {
    document.getElementById('epoch-text').innerText = 'Epoch ' + (batch + 1) + ' Accuracy ' +  logs.loss
}
function onBatchEnd(batch, logs) {
    document.getElementById('batch-text').innerText = 'Batch ' + batch + ' Accuracy ' +  logs.loss
}

function saveModel() {
    model.save('downloads://rpomodel')
}

async function loadModel() {
    let inputEl = document.getElementById('model-input')
    model = await tf.loadLayersModel(
        tf.io.browserFiles([inputEl.files[1], inputEl.files[0]]));
}

function predictModel() {
    let rad = Number(document.getElementById('radial-pred').value)
    let it =  Number(document.getElementById('it-pred').value)

    const {inputMax, inputMin, labelMax, labelMin} = tensorData
    let x = tf.tensor([[rad, it, 1]])
    let normx = x.sub(inputMin).div(inputMax.sub(inputMin))
    let pred = model.predict(normx)
    let unnormpred = pred.mul(labelMax.sub(labelMin)).add(labelMin)
    document.getElementById('radial-pred').value = unnormpred.arraySync()[0][0]
    document.getElementById('it-pred').value = unnormpred.arraySync()[0][1]
}

function startTraining() {
    run(data, model)
}
// run(data, model)