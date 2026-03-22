package red.vortx.neural.util;

public final class DigitRenderer {

    private DigitRenderer() {
    }

    public static String render(double[] pixels, int columns) {
        StringBuilder builder = new StringBuilder();

        for (int index = 0; index < pixels.length; index++) {
            if (index % columns == 0) {
                builder.append('\n');
            }

            double intensity = pixels[index];
            if (intensity < 0.1) {
                builder.append("  ");
            } else if (intensity < 0.5) {
                builder.append("▒ ");
            } else if (intensity < 0.8) {
                builder.append("▓ ");
            } else {
                builder.append("█ ");
            }
        }

        return builder.toString();
    }

}
