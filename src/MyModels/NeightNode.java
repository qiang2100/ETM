package MyModels;

import java.util.ArrayList;
import java.util.HashMap;

public class NeightNode {

	
	HashMap<Integer,ArrayList<Integer>> nei;
	
	public NeightNode()
	{
		nei = new HashMap<Integer, ArrayList<Integer>>();
	}
	
	public void add(int id, int ind)
	{
		if(nei.containsKey(id))
		{
			nei.get(id).add(ind);
		}else
		{
			ArrayList<Integer> neiS = new ArrayList<Integer>();
			neiS.add(ind);
			nei.put(id, neiS);
		}
	}
	
	public ArrayList<Integer> getNei(int id)
	{
		if(!nei.containsKey(id))
			return null;//System.out.println("wrong");
		else
			return nei.get(id);
		
		//return null;
	}
}
