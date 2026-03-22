package red.vortx.neural.io;

import red.vortx.neural.data.MnistImageSet;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

public final class MnistParser {

    public static final int LABEL_MAGIC = 2049;
    public static final int IMAGE_MAGIC = 2051;

    private MnistParser() {
    }

    public static int[] readLabels(InputStream inputStream) throws IOException {
        try (DataInputStream dataInput = new DataInputStream(new BufferedInputStream(inputStream))) {
            int magicNumber = dataInput.readInt();
            if (magicNumber != LABEL_MAGIC) {
                throw new IOException("Invalid label file magic number: " + magicNumber);
            }

            int itemCount = dataInput.readInt();
            int[] labels = new int[itemCount];
            for (int index = 0; index < itemCount; index++) {
                labels[index] = dataInput.readUnsignedByte();
            }
            return labels;
        }
    }

    public static MnistImageSet readImages(InputStream inputStream) throws IOException {
        try (DataInputStream dataInput = new DataInputStream(new BufferedInputStream(inputStream))) {
            int magicNumber = dataInput.readInt();
            if (magicNumber != IMAGE_MAGIC) {
                throw new IOException("Invalid image file magic number: " + magicNumber);
            }

            int imageCount = dataInput.readInt();
            int rows = dataInput.readInt();
            int columns = dataInput.readInt();
            int pixelCount = rows * columns;

            double[][] images = new double[imageCount][pixelCount];
            for (int index = 0; index < imageCount; index++) {
                for (int pixel = 0; pixel < pixelCount; pixel++) {
                    images[index][pixel] = dataInput.readUnsignedByte() / 255.0;
                }
            }

            return new MnistImageSet(images, rows, columns);
        }
    }

}
