package MyModels;

public class Neighbor implements Comparable {

	public int id;
	
	public float sis;
	
	public Neighbor(int id,float sis)
	{
		this.id = id;
		
		this.sis = sis;
	}

	

	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		return   (int)(((Neighbor)o).sis*10000) - (int)(this.sis*10000);
	}
}
