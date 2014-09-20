package csc301.exercise1;

import org.junit.Test;

public class TestTrainCompany {

	// An example of how to verify that an exception is thrown
	@Test(expected=IllegalArgumentException.class)
	public void cannotCreateCompanyWithNullName() {
		new TrainCompany(null);
	}
	
}
