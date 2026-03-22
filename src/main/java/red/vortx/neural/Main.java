package red.vortx.neural;

import org.jspecify.annotations.Nullable;
import red.vortx.neural.data.TrainingSample;
import red.vortx.neural.io.MnistDataLoader;
import red.vortx.neural.io.ModelStorage;
import red.vortx.neural.math.Matrix;
import red.vortx.neural.util.DigitRenderer;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Main {

    private static final int DIGIT_COUNT = 10;
    private static final int HIDDEN_NODES = 64;
    private static final int DEFAULT_EPOCHS = 5;
    private static final double LEARNING_RATE = 0.1;
    private static final String MODEL_FILE = "mnist_brain.bin";

    private static final MnistDataLoader dataLoader = new MnistDataLoader(DIGIT_COUNT);

    private static @Nullable NeuralNetwork brain;
    private static @Nullable List<TrainingSample> trainingData;
    private static MnistDataLoader.@Nullable TestDataSet testData;

    public static void main(String[] args) {
        //vibeslop main class

        Scanner scanner = new Scanner(System.in);
        boolean running = true;

        System.out.println("Neural Network MNIST Manager Initialized.");

        while (running) {
            printMenu();
            String input = scanner.nextLine().trim();

            try {
                switch (input) {
                    case "1" -> train(scanner);
                    case "2" -> testAccuracy();
                    case "3" -> demo();
                    case "4" -> saveModel(scanner);
                    case "5" -> loadModel(scanner);
                    case "6" -> failedDemo();
                    case "0" -> running = false;
                    default -> System.out.println("Unknown command.");
                }
            } catch (Exception exception) {
                System.err.println("Operation failed: " + exception.getMessage());
            }
        }

        scanner.close();
    }

    private static void train(Scanner scanner) throws IOException {
        if (trainingData == null) {
            System.out.println("Loading training data...");
            trainingData = dataLoader.loadTrainingData();
        }

        NeuralNetwork network = getOrCreateBrain();

        System.out.print("Enter epochs (empty for " + DEFAULT_EPOCHS + "): ");
        String line = scanner.nextLine().trim();
        int epochs = line.isEmpty() ? DEFAULT_EPOCHS : Integer.parseInt(line);

        System.out.println("Training...");
        network.train(trainingData, epochs);
        System.out.println("Training session finished.");
    }

    private static void testAccuracy() throws IOException {
        MnistDataLoader.TestDataSet data = getOrLoadTestData();
        NeuralNetwork network = getOrCreateBrain();

        double[][] images = data.images();
        int[] labels = data.labels();

        int correctCount = 0;
        for (int index = 0; index < images.length; index++) {
            Matrix input = Matrix.columnVector(images[index]);
            int prediction = network.predict(input);
            if (prediction == labels[index]) {
                correctCount++;
            }
        }

        double accuracy = (double) correctCount / images.length * 100;
        System.out.printf("Accuracy: %.2f%% (%d/%d)%n", accuracy, correctCount, images.length);
    }

    private static void demo() throws IOException {
        MnistDataLoader.TestDataSet data = getOrLoadTestData();
        NeuralNetwork network = getOrCreateBrain();

        double[][] images = data.images();
        int[] labels = data.labels();

        int index = (int) (Math.random() * images.length);
        double[] pixels = images[index];

        System.out.println(DigitRenderer.render(pixels, data.imageColumns()));

        Matrix input = Matrix.columnVector(pixels);
        int prediction = network.predict(input);

        System.out.println("Expected: " + labels[index]);
        System.out.println("Network:  " + prediction);
    }

    private static void failedDemo() throws IOException {
        MnistDataLoader.TestDataSet data = getOrLoadTestData();
        NeuralNetwork network = getOrCreateBrain();

        double[][] images = data.images();
        int[] labels = data.labels();

        int totalImages = images.length;
        int failuresToFind = 5;
        int foundCount = 0;

        System.out.println("Searching for misclassifications...");

        int offset = (int) (Math.random() * totalImages);

        for (int i = 0; i < totalImages && foundCount < failuresToFind; i++) {
            int currentIndex = (i + offset) % totalImages;
            double[] pixels = images[currentIndex];
            int actualLabel = labels[currentIndex];

            Matrix input = Matrix.columnVector(pixels);
            int prediction = network.predict(input);

            if (prediction != actualLabel) {
                System.out.println(DigitRenderer.render(pixels, data.imageColumns()));
                System.out.printf("Sample #%d | Expected: %d | Predicted: %d%n",
                        currentIndex, actualLabel, prediction
                );
                foundCount++;
            }
        }

        if (foundCount == 0) {
            System.out.println("All correct.");
        }
    }

    private static void saveModel(Scanner scanner) throws IOException {
        NeuralNetwork network = getOrCreateBrain();

        System.out.print("Enter model file (empty for " + MODEL_FILE + "): ");
        String line = scanner.nextLine().trim();
        String filePath = line.isEmpty() ? MODEL_FILE : line;

        System.out.println("Saving model to " + filePath + "...");
        ModelStorage.save(network, filePath);
        System.out.println("Model saved.");
    }

    private static void loadModel(Scanner scanner) throws IOException {
        NeuralNetwork network = getOrCreateBrain();

        System.out.print("Enter model file (empty for " + MODEL_FILE + "): ");
        String line = scanner.nextLine().trim();
        String filePath = line.isEmpty() ? MODEL_FILE : line;

        System.out.println("Loading model from " + filePath + "...");
        ModelStorage.load(network, filePath);
        System.out.println("Model loaded.");
    }

    private static NeuralNetwork getOrCreateBrain() throws IOException {
        if (brain == null) {
            int inputSize = dataLoader.determineInputSize();
            brain = new NeuralNetwork(inputSize, HIDDEN_NODES, DIGIT_COUNT, LEARNING_RATE);
        }
        return brain;
    }

    private static MnistDataLoader.TestDataSet getOrLoadTestData() throws IOException {
        if (testData == null) {
            System.out.println("Loading test data...");
            testData = dataLoader.loadTestData();
        }
        return testData;
    }

    private static void printMenu() {
        System.out.println("\n");
        System.out.println("[1] Train  [2] Test Accuracy  [3] Random Demo");
        System.out.println("[4] Save   [5] Load           [6] Failed Demo");
        System.out.println("           [0] Exit");
        System.out.print("> ");
    }

}
