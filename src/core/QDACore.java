package core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

public class QDACore {

	final Map<Double, MatrixComputation> map = new HashMap<Double, MatrixComputation>();

	public void extractPercentualMatrix(double percent,
			SimpleMatrix originalMatrix, MatrixComputation percentComputation,
			MatrixComputation restComputation) {

		assert percent >= 0 && percent <= 1;

		Random random = new Random((originalMatrix.numRows() * 21) % 37);

		int originalMatrixRowCount = originalMatrix.numRows();

		Set<Integer> catchIndexes = new HashSet<Integer>();
		Set<Integer> restIndexes = new HashSet<Integer>();

		while (catchIndexes.size() / (double) originalMatrixRowCount < percent) {

			Integer integer = new Integer(
					random.nextInt(originalMatrixRowCount));

			if (catchIndexes.contains(integer)) {
				continue;
			}

			catchIndexes.add(integer);
		}

		for (int i = 0; i < originalMatrixRowCount; i = i + 1) {
			restIndexes.add(new Integer(i));
		}

		restIndexes.removeAll(catchIndexes);

		SimpleMatrix percentualMatrix = populateMatrixFromIndexSet(
				originalMatrix, catchIndexes);

		SimpleMatrix restMatrix = populateMatrixFromIndexSet(originalMatrix,
				restIndexes);

		percentComputation.compute(percentualMatrix);
		restComputation.compute(restMatrix);

	}

	private SimpleMatrix populateMatrixFromIndexSet(
			SimpleMatrix originalMatrix, Set<Integer> indexSet) {

		SimpleMatrix newMatrix = new SimpleMatrix(indexSet.size(),
				originalMatrix.numCols());

		int i = 0;
		for (Integer integer : indexSet) {
			newMatrix.insertIntoThis(i, 0,
					originalMatrix.extractVector(true, integer));

			i = i + 1;
		}

		return newMatrix;
	}

	public SimpleMatrix extractMatrixOfClass(SimpleMatrix originalMatrix,
			double classIndex) {

		Set<Integer> indexSet = new HashSet<Integer>();

		for (int rowIndex = 0; rowIndex < originalMatrix.numRows(); rowIndex = rowIndex + 1) {

			if (originalMatrix.get(rowIndex, originalMatrix.numCols() - 1) != classIndex) {
				continue;
			}

			indexSet.add(new Integer(rowIndex));
		}

		return populateMatrixFromIndexSet(originalMatrix, indexSet);
	}

	public List<SimpleMatrix> extractFeaturesVectorsOf(
			SimpleMatrix originalMatrix) {

		SimpleMatrix matrix = originalMatrix.extractMatrix(0,
				originalMatrix.numRows(), 0, originalMatrix.numCols() - 1);

		List<SimpleMatrix> vectors = new LinkedList<SimpleMatrix>();

		for (int i = 0; i < matrix.numRows(); i = i + 1) {
			vectors.add(matrix.extractVector(true, i));
		}

		return vectors;
	}

	public SimpleMatrix mean(List<SimpleMatrix> vectors) {

		SimpleMatrix mean = new SimpleMatrix(1, vectors.get(0).numCols());

		for (SimpleMatrix vector : vectors) {

			mean = mean.plus(vector);
		}

		return mean.scale(1 / (double) vectors.size());
	}

	public SimpleMatrix varianceMatrix(List<SimpleMatrix> vectors) {

		SimpleMatrix mean = mean(vectors);
		SimpleMatrix variance = new SimpleMatrix(mean.numCols(), mean.numCols());

		for (SimpleMatrix vector : vectors) {

			SimpleMatrix difference = vector.minus(mean);

			SimpleMatrix mult = difference.transpose().mult(difference);
			variance = variance.plus(mult);
		}

		return variance.scale(1 / (vectors.size() - (double) 1));
	}

	public MatrixComputation discriminantOfClass(SimpleMatrix originalMatrix,
			double classIndex) {

		List<SimpleMatrix> vectors = extractFeaturesVectorsOf(extractMatrixOfClass(
				originalMatrix, classIndex));

		final SimpleMatrix mean = mean(vectors);

		SimpleMatrix varianceMatrix = varianceMatrix(vectors);

		final double pi = Math.log10(vectors.size()
				/ (double) originalMatrix.numRows());

		final double determinant = Math.log10(varianceMatrix.determinant());

		final SimpleMatrix inverse = varianceMatrix.invert();

		final double independentFromPredictionRequest = pi - .5 * determinant;

		return new MatrixComputation() {

			@Override
			public Double compute(SimpleMatrix predictionRequest) {

				SimpleMatrix difference = predictionRequest.minus(mean);
				SimpleMatrix mult = difference.mult(inverse).mult(
						difference.transpose());

				assert mult.numRows() == 1 && mult.numCols() == 1;

				return independentFromPredictionRequest - .5 * mult.get(0, 0);
			}
		};

	}

	private class MatrixHolder {

		private SimpleMatrix matrix;

		public void hold(SimpleMatrix matrix) {
			this.matrix = matrix;

		}
	}

	public SimpleMatrix learn(SimpleMatrix matrix, double d) {

		final MatrixHolder holder = new MatrixHolder();
		this.extractPercentualMatrix(d, matrix, new MatrixComputation() {

			@Override
			public Double compute(SimpleMatrix matrix) {

				Set<Double> classificationIndexes = findClassificationIndexes(matrix);

				for (Double classIndex : classificationIndexes) {

					QDACore.this.map.put(classIndex, QDACore.this
							.discriminantOfClass(matrix, classIndex));
				}

				return null;

			}
		}, new MatrixComputation() {

			@Override
			public Double compute(SimpleMatrix matrix) {

				holder.hold(matrix);

				return null;
			}
		});

		return holder.matrix;

	}

	public Set<Double> findClassificationIndexes(SimpleMatrix matrix) {

		Set<Double> classes = new HashSet<Double>();

		for (int i = 0; i < matrix.numRows(); i = i + 1) {

			classes.add(matrix.get(i, matrix.numCols() - 1));
		}

		return classes;
	}

	public double predict(SimpleMatrix vector) {

		Double maxPrediction = Double.NEGATIVE_INFINITY;
		Double classIndex = -1d;
		for (Entry<Double, MatrixComputation> entry : map.entrySet()) {
			Double currentPrediction = entry.getValue().compute(vector);
			if (currentPrediction > maxPrediction) {
				maxPrediction = currentPrediction;
				classIndex = entry.getKey();
			}
		}

		return classIndex;
	}
}
