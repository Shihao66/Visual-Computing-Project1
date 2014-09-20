package csc301.exercise1;

public class DirectRoute {
	
	
	public DirectRoute(TrainCompany trainCompany, String fromStation, String toStation, double price) {
	}


	public TrainCompany getTrainCompany() {
		return null;
	}
	
	public void setTrainCompany(TrainCompany trainCompany) {
	}
	
	
	public String getFromStation() {
		return null;
	}
	
	public void setFromStation(String fromStation) {
	}


	public String getToStation() {
		return null;
	}
	
	public void setToStation(String toStation) {
	}


	public double getPrice() {
		return 0;
	}
	
	public void setPrice(double price) {
	}
	
	
	@Override
	public boolean equals(Object obj) {
		return false;
	}
	
	
	@Override
	public String toString() {
		return String.format("%s from %s to %s, %.2f$", getTrainCompany().getName(), 
				getFromStation(), getToStation(), getPrice());
	}
}
