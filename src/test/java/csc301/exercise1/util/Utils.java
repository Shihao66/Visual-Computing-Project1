package csc301.exercise1.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import csc301.exercise1.TrainCompany;

public class Utils {

	
	/**
	 * A utility method that creates and returns a TrainCompany isntance based
	 * on data in a text file.
	 * 
	 * Pre-condition: There is a file called <code>dataFileName</code> in the src/test/resources folder.
	 * 
	 * See existing files in src/test/resources for an example of the format. 
	 */
	public static TrainCompany createCopmanyFromDataFile(String dataFileName) throws IOException{
		InputStream in = ClassLoader.getSystemResourceAsStream(dataFileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		
		// The first line of the file should contain the company's name
		String line = br.readLine();
		TrainCompany company = new TrainCompany(line.trim());
		
		// The next lines are expected to be of the format fromStation,toStation,price
		line = br.readLine();
		while(line != null){
			line = line.trim();
			// Skip blank lines
			if(line.isEmpty()){
				continue;
			}
			
			// Parse the line, and add the route to the company 
			String[] parts = line.split(",");
			company.createOrUpdateDirectRoute(parts[0].trim(), parts[1].trim(), Double.parseDouble(parts[2].trim()));
			
			line = br.readLine();
		}
		
		br.close();
		return company;
	}
	
	
}
