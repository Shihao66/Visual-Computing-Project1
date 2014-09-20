package csc301.exercise1;

import static org.junit.Assert.*;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import csc301.exercise1.util.Constants;
import csc301.exercise1.util.Utils;

public class TestMyTripAdvisor {

	
	/*
	 * You can use the starter code as an example of how to write tests
	 * a bit more efficiently, and cleanly. 
	 * You don't have to use our technique, we are just suggesting something that
	 * you might find it useful.
	 * 
	 * Here's a description of what we do:
	 * 1. We store a train-company's data in a text file.
	 * 2. Before each test, we load all of the data from the files to local variables.
	 *    Note: This is inefficient, and would be an issue in a project that has many 
	 *    tests and data files. For this exercise, we chose to not worry about this issue.
	 * 
	 * Technical notes:
	 * - All data files are stored in src/test/resources.
	 *   This is important! If you data files are not stored with the project, we will
	 *   not be able to use them, when running your tests.
	 * - We've defined some utility classes in src/test/java/csc301/exercise1/util.
	 *   Feel free to add/modify code in these classes.
	 * - You might find it useful to come up with a few more utility functions that
	 *   will help you write cleaner tests.
	 *   
	 * TODO: Delete this comment before you submit the assignment.
	 */
	
	
	private TrainCompany fastTrain;
	private TrainCompany swiftRail;


	@Before
	public void setUp() throws Exception {
		// Create TrainCompany instances from data files in the resources folder.
		fastTrain = Utils.createCopmanyFromDataFile("FastTrain.txt");
		swiftRail = Utils.createCopmanyFromDataFile("SwiftRail.txt");
	}

	
	
	
	/*
	 * We added two examples of a tests, feel free to edit/remove them.
	 * 
	 * Notice how we set a timeout of 3000ms - If the trip advisor that we're testing
	 * is broken, and goes into an infinite loop, our tests will not get stuck, they
	 * will just fail after 3 seconds.
	 * 
	 * TODO: Delete this comment before you submit the assignment.
	 */
	
	@Test(timeout=3000)
	public void twoRouteTripWhereRoutesAreFromDifferentCompanies() {
		MyTripAdvisor advisor = new MyTripAdvisor();
		advisor.addTrainCopmany(fastTrain);
		advisor.addTrainCopmany(swiftRail);
		
		List<DirectRoute> trip = advisor.getCheapestTrip(Constants.TORONTO, Constants.MONTREAL);
		
		// Make sure that we got the route we expect
		assertEquals(2, trip.size());
		assertEquals(new DirectRoute(swiftRail, Constants.TORONTO, Constants.OTTAWA, 30), trip.get(0));
		assertEquals(new DirectRoute(fastTrain, Constants.OTTAWA, Constants.MONTREAL, 25), trip.get(1));
	}
	
	
	@Test(timeout=3000)
	public void priceOfTwoRouteTripWhereRoutesAreFromDifferentCompanies() {
		MyTripAdvisor advisor = new MyTripAdvisor();
		advisor.addTrainCopmany(fastTrain);
		advisor.addTrainCopmany(swiftRail);
		
		// Make sure that we got the total price we expect
		assertTrue(55 == advisor.getCheapestPrice(Constants.TORONTO, Constants.MONTREAL));
	}

}
