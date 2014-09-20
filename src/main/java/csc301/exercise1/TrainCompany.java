package csc301.exercise1;

import java.util.Collection;

public class TrainCompany {

	
	public TrainCompany(String name) {
	}
	
	
	
	@Override
	public String toString() {
		return String.format("%s, offering %d routes between %d stations", 
				getName(), getDirectRoutesCount(), getStationsCount());
	}
	
	
	
	public String getName() {
		return null;
	}
	
	public void setName(String name) {
	}
	
	
	
	/**
	 * @return The DirectRoute object that was created/updated.
	 */
	public DirectRoute createOrUpdateDirectRoute(String fromStation, String toStation, double price){
		return null;
	}
	
	
	/**
	 * Delete the specified route, if it exists.
	 */
	public void deleteDirectRoute(String fromStation, String toStation){
		
	}
	
	
	/**
	 * @return null if there is no route from <code>fromStation</code> to
	 * 			<code>toStation</code> with this TrainCompany.
	 */
	public DirectRoute getDirectRoute(String fromStation, String toStation){
		return null;
	}
	
	public Collection<DirectRoute> getDirectRoutesFrom(String fromStation){
		return null;
	}
	
	public Collection<DirectRoute> getRoutesTo(String toStation){
		return null;
	}
	
	public Collection<DirectRoute> getAllDirectRoutes(){
		return null;
	}
	
	
	
	public int getDirectRoutesCount(){
		return getAllDirectRoutes().size();
	}
	
	
	/**
	 * @return The number of stations with service by this TrainCompany.
	 * To be clearer:
	 * - Take the union of all stations (from and to) from this.getAllDirectRoutes()
	 * - Count the unique number of stations (i.e. You only count a station
	 *   once, even if there are multiple routes from/to this station) 
	 */
	public int getStationsCount(){
		return 0;
	}
}
