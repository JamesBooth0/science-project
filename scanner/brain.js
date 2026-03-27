class NeuralNetwork {
    constructor(inputs, hidden, outputs) {
        this.is = inputs;
        this.hs = hidden;
        this.os = outputs;
        this.lr = 0.1;

        // Initialize Weights
        this.w_ih = Array.from({length: this.hs}, () => Array.from({length: this.is}, () => Math.random() * 2 - 1));
        this.w_ho = Array.from({length: this.os}, () => Array.from({length: this.hs}, () => Math.random() * 2 - 1));
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

    predict(inputArray) {
        let hidden = this.w_ih.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * inputArray[i], 0)));
        let output = this.w_ho.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * hidden[i], 0)));
        return { output, hidden };
    }

    train(inputArray, targetArray) {
        // 1. Feedforward
        let { output, hidden } = this.predict(inputArray);

        // 2. Calculate Output Errors
        let outputErrors = targetArray.map((t, i) => t - output[i]);

        // 3. Backpropagation: Hidden to Output
        for (let i = 0; i < this.w_ho.length; i++) {
            let gradient = output[i] * (1 - output[i]) * outputErrors[i] * this.lr;
            for (let j = 0; j < this.w_ho[i].length; j++) {
                this.w_ho[i][j] += gradient * hidden[j];
            }
        }

        // 4. Backpropagation: Input to Hidden (The missing piece!)
        let hiddenErrors = new Array(this.hs).fill(0);
        for (let i = 0; i < this.os; i++) {
            for (let j = 0; j < this.hs; j++) {
                hiddenErrors[j] += outputErrors[i] * this.w_ho[i][j];
            }
        }

        for (let i = 0; i < this.hs; i++) {
            let gradient = hidden[i] * (1 - hidden[i]) * hiddenErrors[i] * this.lr;
            for (let j = 0; j < this.is; j++) {
                this.w_ih[i][j] += gradient * inputArray[j];
            }
        }
    }
}

// HELPER: Normalize function
function normalize(val, min, max) { return (val - min) / (max - min); }

// INITIALIZE
const brain = new NeuralNetwork(3, 5, 1); 

// TRAINING: 10,000 simulations
for (let i = 0; i < 10000; i++) {
    // Scenario A: Fever (103°F) + Fast Heart (110bpm) + Day 3 = COMPLICATION (1)
    brain.train([normalize(103, 95, 105), normalize(110, 40, 160), normalize(3, 1, 14)], [1]);
    
    // Scenario B: Normal (98.6°F) + Relaxed (70bpm) + Day 1 = HEALTHY (0)
    brain.train([normalize(98.6, 95, 105), normalize(70, 40, 160), normalize(1, 1, 14)], [0]);
}

// TEST A NEW PATIENT
let testTemp = 101.5; // Mild fever
let testHR = 95;      // Slightly elevated
let testDay = 4;      

let finalResult = brain.predict([normalize(testTemp, 95, 105), normalize(testHR, 40, 160), normalize(testDay, 1, 14)]);
console.log(`Risk for Patient (Temp: ${testTemp}, HR: ${testHR}):`, (finalResult.output[0] * 100).toFixed(2) + "%");
