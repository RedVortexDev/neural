package red.vortx.neural.math;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

public class Matrix {

    private final double[][] data;
    private final int rows;
    private final int columns;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new double[rows][columns];
    }

    public static Matrix multiply(Matrix left, Matrix right) {
        if (left.columns != right.rows) {
            throw new IllegalArgumentException("Dimensions mismatch: " + left.columns + " != " + right.rows);
        }
        Matrix result = new Matrix(left.rows, right.columns);
        for (int row = 0; row < left.rows; row++) {
            for (int column = 0; column < right.columns; column++) {
                double sum = 0;
                for (int inner = 0; inner < left.columns; inner++) {
                    sum += left.data[row][inner] * right.data[inner][column];
                }
                result.data[row][column] = sum;
            }
        }
        return result;
    }

    public static Matrix subtract(Matrix left, Matrix right) {
        requireSameDimensions(left, right);
        Matrix result = new Matrix(left.rows, left.columns);
        for (int row = 0; row < left.rows; row++) {
            for (int column = 0; column < left.columns; column++) {
                result.data[row][column] = left.data[row][column] - right.data[row][column];
            }
        }
        return result;
    }

    public static Matrix columnVector(double[] values) {
        Matrix result = new Matrix(values.length, 1);
        for (int row = 0; row < values.length; row++) {
            result.data[row][0] = values[row];
        }
        return result;
    }

    private static void requireSameDimensions(Matrix left, Matrix right) {
        if (left.rows != right.rows || left.columns != right.columns) {
            throw new IllegalArgumentException(
                    "Dimensions mismatch: " + left.rows + "x" + left.columns + " vs " + right.rows + "x" + right.columns
            );
        }
    }

    public Matrix transpose() {
        Matrix result = new Matrix(columns, rows);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                result.data[column][row] = this.data[row][column];
            }
        }
        return result;
    }

    public void add(Matrix other) {
        requireSameDimensions(this, other);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                this.data[row][column] += other.data[row][column];
            }
        }
    }

    public void multiplyElementwise(Matrix other) {
        requireSameDimensions(this, other);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                this.data[row][column] *= other.data[row][column];
            }
        }
    }

    public void scale(double scalar) {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                this.data[row][column] *= scalar;
            }
        }
    }

    public double sum() {
        double total = 0;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                total += this.data[row][column];
            }
        }
        return total;
    }

    public void map(ElementOperation operation) {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                this.data[row][column] = operation.apply(this.data[row][column]);
            }
        }
    }

    public void randomize() {
        Random random = new Random();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                this.data[row][column] = random.nextGaussian();
            }
        }
    }

    public void set(int row, int column, double value) {
        this.data[row][column] = value;
    }

    public double get(int row, int column) {
        return this.data[row][column];
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public Matrix copy() {
        Matrix duplicate = new Matrix(rows, columns);
        for (int row = 0; row < rows; row++) {
            System.arraycopy(this.data[row], 0, duplicate.data[row], 0, columns);
        }
        return duplicate;
    }

    public void save(DataOutputStream outputStream) throws IOException {
        outputStream.writeInt(rows);
        outputStream.writeInt(columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                outputStream.writeDouble(data[row][column]);
            }
        }
    }

    public void load(DataInputStream inputStream) throws IOException {
        int savedRows = inputStream.readInt();
        int savedColumns = inputStream.readInt();
        if (savedRows != rows || savedColumns != columns) {
            throw new IOException(
                    "Matrix dimension mismatch: expected " + rows + "x" + columns + " but found " + savedRows + "x" + savedColumns
            );
        }
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                data[row][column] = inputStream.readDouble();
            }
        }
    }

    public interface ElementOperation {

        double apply(double value);

    }

}
