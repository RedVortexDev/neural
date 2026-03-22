package red.vortx.neural.io;

import red.vortx.neural.NeuralNetwork;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public final class ModelStorage {

    private ModelStorage() {
    }

    public static void save(NeuralNetwork network, String filePath) throws IOException {
        try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            network.getInputHiddenWeights().save(output);
            network.getHiddenOutputWeights().save(output);
            network.getHiddenBiases().save(output);
            network.getOutputBiases().save(output);
        }
    }

    public static void load(NeuralNetwork network, String filePath) throws IOException {
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            network.getInputHiddenWeights().load(input);
            network.getHiddenOutputWeights().load(input);
            network.getHiddenBiases().load(input);
            network.getOutputBiases().load(input);
        }
    }

}
