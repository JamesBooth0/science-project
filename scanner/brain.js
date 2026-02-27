// This code is 100% written by AI

class NeuralNetwork {
    constructor(inputs, hidden, outputs) {
        this.is = inputs;
        this.hs = hidden;
        this.os = outputs;

        // Weights: Random numbers between -1 and 1
        this.w_ih = Array.from({length: this.hs}, () => Array.from({length: this.is}, () => Math.random() * 2 - 1));
        this.w_ho = Array.from({length: this.os}, () => Array.from({length: this.hs}, () => Math.random() * 2 - 1));
        this.lr = 0.1; // Learning Rate
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

    predict(inputArray) {
        let hidden = this.w_ih.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * inputArray[i], 0)));
        let output = this.w_ho.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * hidden[i], 0)));
        return output;
    }

    train(inputArray, targetArray) {
        // 1. Feedforward
        let hidden = this.w_ih.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * inputArray[i], 0)));
        let outputs = this.w_ho.map(row => this.sigmoid(row.reduce((acc, w, i) => acc + w * hidden[i], 0)));

        // 2. Calculate Errors
        let outputErrors = targetArray.map((t, i) => t - outputs[i]);

        // 3. Backpropagation (Hidden to Output)
        for (let i = 0; i < this.w_ho.length; i++) {
            for (let j = 0; j < this.w_ho[i].length; j++) {
                let gradient = outputs[i] * (1 - outputs[i]);
                this.w_ho[i][j] += this.lr * outputErrors[i] * gradient * hidden[j];
            }
        }
        return outputErrors[0]; // Return error for tracking
    }
}
