package core;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class CoreTest {

	@Test
	public void produce_a_random_matrix() {

		QDACore core = new QDACore();
		final double percent = .80;
		final int numRows = 2000;
		SimpleMatrix originalMatrix = SimpleMatrix.random(numRows, 40, 0, 100,
				new Random());

		MatrixComputation percentComputation = new MatrixComputation() {

			@Override
			public Double compute(SimpleMatrix matrix) {
				Assert.assertTrue(matrix.numRows() >= percent * numRows);
				return null;
			}
		};

		MatrixComputation restComputation = new MatrixComputation() {

			@Override
			public Double compute(SimpleMatrix matrix) {
				return null;
			}
		};

		core.extractPercentualMatrix(percent, originalMatrix,
				percentComputation, restComputation);
	}

	@Test
	public void extractSubMatrixOfClassK() {

		QDACore core = new QDACore();
		double zero_zero = 1.45;
		double zero_one = 1.23;
		double zero_two = 1;
		SimpleMatrix originalMatrix = new SimpleMatrix(new double[][] {
				{ zero_zero, zero_one, zero_two }, { 2.45, 2.23, 2 },
				{ 3.45, 3.23, 3 } });

		SimpleMatrix matrix = core.extractMatrixOfClass(originalMatrix,
				zero_two);

		assertThat(matrix.numRows(), is(new Integer(1)));
		assertThat(matrix.extractVector(true, 0).numCols(), is(new Integer(3)));
		assertThat(matrix.get(0, 0), is(zero_zero));
		assertThat(matrix.get(0, 1), is(zero_one));
		assertThat(matrix.get(0, 2), is(zero_two));

	}

	@Test
	public void extractFeatureVectorsOfSubMatrixOfClassK() {

		QDACore core = new QDACore();
		double zero_zero = 1.45;
		double zero_one = 1.23;
		double zero_two = 1;
		SimpleMatrix originalMatrix = new SimpleMatrix(new double[][] {
				{ zero_zero, zero_one, zero_two }, { 2.45, 2.23, 2 },
				{ 3.45, 3.23, 3 } });

		List<SimpleMatrix> vectors = core.extractFeaturesVectorsOf(core
				.extractMatrixOfClass(originalMatrix, zero_two));

		assertThat(vectors.size(), is(new Integer(1)));

		SimpleMatrix vector = vectors.get(0);

		assertThat(vector.numCols(), is(new Integer(2)));
		assertThat(vector.get(0, 0), is(zero_zero));
		assertThat(vector.get(0, 1), is(zero_one));

	}

	@Test
	public void computeMeanVector() {

		QDACore core = new QDACore();

		double zero_two = 1;
		SimpleMatrix originalMatrix = new SimpleMatrix(new double[][] {
				{ 1, 2, zero_two }, { 3, 4, 1 }, { 3.45, 3.23, 3 } });

		List<SimpleMatrix> vectors = core.extractFeaturesVectorsOf(core
				.extractMatrixOfClass(originalMatrix, zero_two));

		SimpleMatrix meanVector = core.mean(vectors);

		assertThat(meanVector.numRows(), is(new Integer(1)));
		assertThat(meanVector.numCols(), is(new Integer(2)));
		assertThat(meanVector.get(0, 0), is(new Double(2)));
		assertThat(meanVector.get(0, 1), is(new Double(3)));

	}

	@Test
	public void computeVarianceMatrix() {

		QDACore core = new QDACore();

		double classIndex = 1;
		SimpleMatrix originalMatrix = new SimpleMatrix(new double[][] {
				{ 1, 2, classIndex }, { 3, 4, classIndex }, { 3.45, 3.23, 3 } });

		List<SimpleMatrix> vectors = core.extractFeaturesVectorsOf(core
				.extractMatrixOfClass(originalMatrix, classIndex));

		SimpleMatrix varianceMatrix = core.varianceMatrix(vectors);

		assertThat(varianceMatrix.numRows(), is(new Integer(2)));
		assertThat(varianceMatrix.numCols(), is(new Integer(2)));
		assertThat(varianceMatrix.get(0, 0), is(new Double(2)));
		assertThat(varianceMatrix.get(0, 1), is(new Double(2)));
		assertThat(varianceMatrix.get(1, 0), is(new Double(2)));
		assertThat(varianceMatrix.get(1, 1), is(new Double(2)));

	}

	@Test
	public void computeDiscriminantValueForSomeClass() {

		QDACore core = new QDACore();

		double classIndex = 1;
		SimpleMatrix originalMatrix = new SimpleMatrix(new double[][] {
				{ .98, 2.756, classIndex }, { 1.345, 3.34, classIndex },
				{ 3.45, 3.23, 3 } });

		MatrixComputation discriminantComputation = core.discriminantOfClass(
				originalMatrix, classIndex);

		double discriminant = discriminantComputation.compute(new SimpleMatrix(
				new double[][] { { .99991, 2.0001 } }));

	}

	@Test
	public void readingLetterDataset() throws IOException {

		SimpleMatrix matrix = loadMatrixFromLetterDataset();

		Assert.assertEquals(20000, matrix.numRows());
	}

	@Test
	public void test_prediction_up_to_eighty_percent()
			throws FileNotFoundException, IOException {

		QDACore core = new QDACore();
		SimpleMatrix matrix = loadMatrixFromLetterDataset();
		SimpleMatrix testMatrix = core.learn(matrix, .80);

		int correct = 0;

		for (int i = 0; i < testMatrix.numRows(); i = i + 1) {

			SimpleMatrix vector = testMatrix.extractVector(true, i);
			double predictedClass = core.predict(vector.extractMatrix(0, 1, 0,
					vector.numCols() - 1));

			if (predictedClass == vector.get(0, vector.numCols() - 1)) {
				correct = correct + 1;
			}
		}

		Assert.assertTrue((double) correct / testMatrix.numRows() > .80);
	}

	@Test
	public void findClasses() throws FileNotFoundException, IOException {

		QDACore core = new QDACore();
		Set<Double> classes = core
				.findClassificationIndexes(loadMatrixFromLetterDataset());

		for (Double classIndex : classes) {
			Assert.assertTrue(classIndex >= 0 && classIndex <= 27);
		}

	}

	private SimpleMatrix loadMatrixFromLetterDataset()
			throws FileNotFoundException, IOException {
		File lettersDataset = new File("letter-recognition.data");

		FileInputStream inputStream = new FileInputStream(lettersDataset);
		DataInputStream in = new DataInputStream(inputStream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		String strLine;
		List<String[]> learningRows = new LinkedList<String[]>();
		// Read File Line By Line
		while ((strLine = br.readLine()) != null) {
			String[] splitted = strLine.split(",");
			learningRows.add(splitted);
		}
		// Close the input stream
		in.close();

		double[][] matrix = new double[learningRows.size()][learningRows.get(0).length];

		int rowIndex = 0;
		for (String[] vector : learningRows) {

			for (int i = 1; i < vector.length; i = i + 1) {
				matrix[rowIndex][i - 1] = new Double(vector[i]);
			}

			matrix[rowIndex][vector.length - 1] = new Double(
					vector[0].charAt(0) - 'A');

			rowIndex = rowIndex + 1;
		}

		return new SimpleMatrix(matrix);
	}
}
