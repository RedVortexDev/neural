import org.junit.jupiter.api.Test;
import red.vortx.neural.math.Matrix;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixTest {

    @Test
    void testMatrixAddition() {
        Matrix matrixA = new Matrix(2, 2);
        matrixA.set(0, 0, 1);
        matrixA.set(0, 1, 2);
        matrixA.set(1, 0, 3);
        matrixA.set(1, 1, 4);

        Matrix matrixB = new Matrix(2, 2);
        matrixB.set(0, 0, 5);
        matrixB.set(0, 1, 6);
        matrixB.set(1, 0, 7);
        matrixB.set(1, 1, 8);

        matrixA.add(matrixB);

        assertEquals(6, matrixA.get(0, 0));
        assertEquals(8, matrixA.get(0, 1));
        assertEquals(10, matrixA.get(1, 0));
        assertEquals(12, matrixA.get(1, 1));
    }

    @Test
    void testMatrixSubtraction() {
        Matrix matrixA = new Matrix(2, 2);
        matrixA.set(0, 0, 5);
        matrixA.set(0, 1, 6);
        matrixA.set(1, 0, 7);
        matrixA.set(1, 1, 8);

        Matrix matrixB = new Matrix(2, 2);
        matrixB.set(0, 0, 1);
        matrixB.set(0, 1, 2);
        matrixB.set(1, 0, 3);
        matrixB.set(1, 1, 4);

        Matrix result = Matrix.subtract(matrixA, matrixB);

        assertEquals(4, result.get(0, 0));
        assertEquals(4, result.get(0, 1));
        assertEquals(4, result.get(1, 0));
        assertEquals(4, result.get(1, 1));
    }

    @Test
    void testMatrixMultiplication() {
        Matrix matrixA = new Matrix(2, 3);
        matrixA.set(0, 0, 1);
        matrixA.set(0, 1, 2);
        matrixA.set(0, 2, 3);
        matrixA.set(1, 0, 4);
        matrixA.set(1, 1, 5);
        matrixA.set(1, 2, 6);

        Matrix matrixB = new Matrix(3, 2);
        matrixB.set(0, 0, 7);
        matrixB.set(0, 1, 8);
        matrixB.set(1, 0, 9);
        matrixB.set(1, 1, 10);
        matrixB.set(2, 0, 11);
        matrixB.set(2, 1, 12);

        Matrix result = Matrix.multiply(matrixA, matrixB);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getColumns());

        // (1*7 + 2*9 + 3*11) = 58
        assertEquals(58, result.get(0, 0));
        // (1*8 + 2*10 + 3*12) = 64
        assertEquals(64, result.get(0, 1));
        // (4*7 + 5*9 + 6*11) = 139
        assertEquals(139, result.get(1, 0));
        // (4*8 + 5*10 + 6*12) = 154
        assertEquals(154, result.get(1, 1));
    }

    @Test
    void testMatrixMultiplicationDimensionMismatch() {
        Matrix matrixA = new Matrix(2, 2);
        Matrix matrixB = new Matrix(3, 2);

        assertThrows(
            IllegalArgumentException.class,
            () -> Matrix.multiply(matrixA, matrixB)
        );
    }

    @Test
    void testTranspose() {
        Matrix matrix = new Matrix(2, 3);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(0, 2, 3);
        matrix.set(1, 0, 4);
        matrix.set(1, 1, 5);
        matrix.set(1, 2, 6);

        Matrix transposed = matrix.transpose();

        assertEquals(3, transposed.getRows());
        assertEquals(2, transposed.getColumns());

        assertEquals(1, transposed.get(0, 0));
        assertEquals(4, transposed.get(0, 1));
        assertEquals(2, transposed.get(1, 0));
        assertEquals(5, transposed.get(1, 1));
        assertEquals(3, transposed.get(2, 0));
        assertEquals(6, transposed.get(2, 1));
    }

    @Test
    void testMap() {
        Matrix matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(1, 0, 3);
        matrix.set(1, 1, 4);

        matrix.map(value -> value * 10);

        assertEquals(10, matrix.get(0, 0));
        assertEquals(20, matrix.get(0, 1));
        assertEquals(30, matrix.get(1, 0));
        assertEquals(40, matrix.get(1, 1));
    }

}
