package red.vortx.neural.data;

import red.vortx.neural.math.Matrix;

public record TrainingSample(Matrix input, Matrix target) {

    public static TrainingSample fromLabeledImage(double[] pixels, int label, int classCount) {
        Matrix input = Matrix.columnVector(pixels);
        Matrix target = createOneHotTarget(label, classCount);
        return new TrainingSample(input, target);
    }

    private static Matrix createOneHotTarget(int labelIndex, int classCount) {
        Matrix target = new Matrix(classCount, 1);
        target.set(labelIndex, 0, 1.0);
        return target;
    }

}
