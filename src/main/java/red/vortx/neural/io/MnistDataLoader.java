package red.vortx.neural.io;

import red.vortx.neural.data.MnistImageSet;
import red.vortx.neural.data.TrainingSample;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MnistDataLoader {

    private static final String TRAIN_IMAGES = "/dataset/train-images.idx3-ubyte";
    private static final String TRAIN_LABELS = "/dataset/train-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "/dataset/t10k-images.idx3-ubyte";
    private static final String TEST_LABELS = "/dataset/t10k-labels.idx1-ubyte";

    private final int classCount;

    public MnistDataLoader(int classCount) {
        this.classCount = classCount;
    }

    public List<TrainingSample> loadTrainingData() throws IOException {
        MnistImageSet imageSet = loadImageSet(TRAIN_IMAGES);
        int[] labels = loadLabels(TRAIN_LABELS);
        double[][] images = imageSet.images();

        List<TrainingSample> samples = new ArrayList<>(images.length);
        for (int index = 0; index < images.length; index++) {
            samples.add(TrainingSample.fromLabeledImage(images[index], labels[index], classCount));
        }
        return samples;
    }

    public TestDataSet loadTestData() throws IOException {
        MnistImageSet imageSet = loadImageSet(TEST_IMAGES);
        int[] labels = loadLabels(TEST_LABELS);
        return new TestDataSet(imageSet.images(), labels, imageSet.columns());
    }

    public int determineInputSize() throws IOException {
        MnistImageSet imageSet = loadImageSet(TRAIN_IMAGES);
        return imageSet.rows() * imageSet.columns();
    }

    private MnistImageSet loadImageSet(String resourcePath) throws IOException {
        InputStream stream = getClass().getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new IOException("Resource not found: " + resourcePath);
        }
        return MnistParser.readImages(stream);
    }

    private int[] loadLabels(String resourcePath) throws IOException {
        InputStream stream = getClass().getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new IOException("Resource not found: " + resourcePath);
        }
        return MnistParser.readLabels(stream);
    }

    public record TestDataSet(double[][] images, int[] labels, int imageColumns) {

    }

}
