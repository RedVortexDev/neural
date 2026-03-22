package red.vortx.neural.math;

public final class Activation {

    private Activation() {
    }

    public static double sigmoid(double x) {
        // 1 / (1 + e^-x)
        return 1 / (1 + Math.exp(-x));
    }

    public static double derivativeSigmoid(double sigmoid) {
        // σ(x) * (1 - σ(x))
        return sigmoid * (1 - sigmoid);
    }

    public static double relu(double x) {
        // max(0, x)
        return Math.max(0, x);
    }

    public static double derivativeRelu(double x) {
        // x > 0 ? 1 : 0
        return x > 0 ? 1.0 : 0.0;
    }

    public static double leakyRelu(double x) {
        // x > 0 ? x : 0.01x
        return x > 0 ? x : 0.01 * x;
    }

    public static double derivativeLeakyRelu(double x) {
        // x > 0 ? 1 : 0.01
        return x > 0 ? 1.0 : 0.01;
    }

}
