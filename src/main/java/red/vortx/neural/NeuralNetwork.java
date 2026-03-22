package red.vortx.neural;

import red.vortx.neural.data.TrainingSample;
import red.vortx.neural.math.Activation;
import red.vortx.neural.math.Matrix;

import java.util.Collections;
import java.util.List;

public class NeuralNetwork {

    private final Matrix inputHiddenWeights;
    private final Matrix hiddenOutputWeights;
    private final Matrix hiddenBiases;
    private final Matrix outputBiases;
    private final double learningRate;

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        this.inputHiddenWeights = new Matrix(hiddenNodes, inputNodes);
        this.hiddenOutputWeights = new Matrix(outputNodes, hiddenNodes);
        this.hiddenBiases = new Matrix(hiddenNodes, 1);
        this.outputBiases = new Matrix(outputNodes, 1);
        this.learningRate = learningRate;

        inputHiddenWeights.randomize();
        hiddenOutputWeights.randomize();
        hiddenBiases.randomize();
        outputBiases.randomize();
    }

    public int predict(Matrix input) {
        Matrix output = feedForward(input).output();
        return indexOfMax(output);
    }

    public FeedForwardResult feedForward(Matrix input) {
        Matrix hidden = Matrix.multiply(inputHiddenWeights, input);
        hidden.add(hiddenBiases);
        hidden.map(Activation::sigmoid);

        Matrix output = Matrix.multiply(hiddenOutputWeights, hidden);
        output.add(outputBiases);
        output.map(Activation::sigmoid);

        return new FeedForwardResult(hidden, output);
    }

    public double trainSingle(Matrix input, Matrix target) {
        FeedForwardResult result = feedForward(input);
        Matrix hidden = result.hidden();
        Matrix output = result.output();

        Matrix outputErrors = Matrix.subtract(target, output);

        Matrix outputGradient = computeGradient(output, outputErrors);
        applyWeightUpdate(hiddenOutputWeights, outputBiases, outputGradient, hidden);

        Matrix hiddenErrors = Matrix.multiply(hiddenOutputWeights.transpose(), outputErrors);
        Matrix hiddenGradient = computeGradient(hidden, hiddenErrors);
        applyWeightUpdate(inputHiddenWeights, hiddenBiases, hiddenGradient, input);

        outputErrors.map(value -> value * value);
        return outputErrors.sum();
    }

    public void train(List<TrainingSample> samples, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(samples);
            double totalLoss = 0;

            for (TrainingSample sample : samples) {
                totalLoss += trainSingle(sample.input(), sample.target());
            }

            double averageLoss = totalLoss / samples.size();
            System.out.printf("Epoch %d complete. Average Loss: %.5f%n", epoch + 1, averageLoss);
        }
    }

    public Matrix getInputHiddenWeights() {
        return inputHiddenWeights;
    }

    public Matrix getHiddenOutputWeights() {
        return hiddenOutputWeights;
    }

    public Matrix getHiddenBiases() {
        return hiddenBiases;
    }

    public Matrix getOutputBiases() {
        return outputBiases;
    }

    private Matrix computeGradient(Matrix layerOutput, Matrix errors) {
        Matrix gradient = layerOutput.copy();
        gradient.map(Activation::derivativeSigmoid);
        gradient.multiplyElementwise(errors);
        gradient.scale(learningRate);
        return gradient;
    }

    private void applyWeightUpdate(Matrix weights, Matrix biases, Matrix gradient, Matrix previousLayerOutput) {
        Matrix weightDeltas = Matrix.multiply(gradient, previousLayerOutput.transpose());
        weights.add(weightDeltas);
        biases.add(gradient);
    }

    private int indexOfMax(Matrix matrix) {
        int bestIndex = 0;
        double bestValue = -1;
        for (int row = 0; row < matrix.getRows(); row++) {
            double value = matrix.get(row, 0);
            if (value > bestValue) {
                bestValue = value;
                bestIndex = row;
            }
        }
        return bestIndex;
    }

    public record FeedForwardResult(Matrix hidden, Matrix output) {

    }

}
