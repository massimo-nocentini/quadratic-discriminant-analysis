package learning;

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

public class LearningEJML {

	@Test
	public void test() {
		SimpleMatrix m = new SimpleMatrix(new double[][] { { 3.2, 1.2 },
				{ 4, 5 } });

		SimpleMatrix extractVector = m.extractVector(true, 0);

	}

}
