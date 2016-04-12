package MyModels;



import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;




public class ETM {

	//String dataName = "20News";
	String dataName = "googlenews";
	 /**
     * topic assignments for each word.
     */
    int z[][];


    /**
     * cwt[i][j] number of instances of word i (term?) assigned to topic j.
     */
    int[][] nw;


    /**
     * na[i][j] number of words in document i assigned to topic j.
     */
    int[][] nd;


    /**
     * nwsum[j] total number of words assigned to topic j.
     */
    int[] nwsum;
   
    /**
     * nasum[i] total number of words in document i.
     */
    int[] ndsum;
  
    /**
     * cumulative statistics of theta
     */
    double[][] thetasum;

 
    /**
     * size of statistics
     */
    int numstats;

    int num_neighbors_foreach_word[];//word - number of neighbors of this word	
	int word_neighbors[][];//word - neighbors of this word
	
    /**
     * sampling lag (?)
     */
    private static int THIN_INTERVAL = 20;

    /**
     * burn-in period
     */
    private static int BURN_IN = 100;

    /**
     * sample lag (if -1 only one sample taken)
     */
    private static int SAMPLE_LAG;

    private static int dispcol = 0;

	/**
     * the number of pseudo-documents
     */

	int numPse;
    /**
     * vocabulary size
     */
    int V;

    /**
     * number of topics
     */
    int K;

    /**
     * Dirichlet parameter (document--topic associations)
     */
    double alpha=.1;

    /**
     * Dirichlet parameter (topic--term associations)
     */
    double beta=.1;
    
    /**
     * topic assignments for each document.
     */
    int ps[];
    
    /**
     * number of documents in cluster z.
     */
    int m_ps[];
    
    /**
     * max iterations
     */
    private static int ITERATIONS = 1000;
    
    
    ArrayList<Integer> lablesArr = new ArrayList<Integer>();
    /**
     * the number of clusters
    */
    
    int clustering;
    
    double[][] phisum;
    
	ArrayList<Integer> label ;
	//ArrayList<String> gene ;
	ArrayList<ArrayList<Integer>> sData ;
	//ArrayList<ArrayList<Integer>> sDataNum;
	
	ArrayList<ArrayList<Integer>> pseData ;
	ArrayList<ArrayList<Integer>> shortIndArr;
	ArrayList<ArrayList<Integer>> longIndArr;
	
	ArrayList<String> wordsArr = new ArrayList<String>();
	
	ArrayList<NeightNode> sentNeighArr;
	//HashMap<String,Integer> wordIdMap = new HashMap<String,Integer>();
	HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	
	double textDist[][];
	
	double wordDist[][];
	
	int clu[];
	
	
	
  
    public ETM(String dn) {
    	dataName = dn;
    }
    public void initialState(int K) {
        //   int i;

           int M = pseData.size();

           // initialise count variables.
           nw = new int[V][K];
           nd = new int[M][K];
           nwsum = new int[K];
           ndsum = new int[M];

           // The z_i are are initialised to values in [1,K] to determine the
           // initial state of the Markov chain.

           z = new int[M][];
           
           //clu = new int[M];
           for (int m = 0; m < M; m++) {
               int N = pseData.get(m).size();
               z[m] = new int[N];
               
               
               
               for (int n = 0; n < N; n++) {
            	   int topic = (int) (Math.random() * K);
                   z[m][n] = topic;
                   // number of instances of word i assigned to topic j
                   nw[pseData.get(m).get(n)][topic]++;
                   // number of words in document i assigned to topic j.
                   nd[m][topic]++;
                   // total number of words assigned to topic j.
                   nwsum[topic]++;
               }
               // total number of words in document i
               ndsum[m] = N;
           }
       }
    
    private int sampleFullConditional(int m, int n) {

        // remove z_i from the count variables
    	
    	double neiProb[] = computNeiProb(m,n);
    	
        int topic = z[m][n];
        nw[pseData.get(m).get(n)][topic]--;
        nd[m][topic]--;
        nwsum[topic]--;
        //ndsum[m]--;

        // do multinomial sampling via cumulative method:
        double[] p = new double[K];
        if(neiProb==null)
        {
	        for (int k = 0; k < K; k++) {
	            p[k] = (nw[pseData.get(m).get(n)][k] + beta) / (nwsum[k] + V * beta)
	                * (nd[m][k] + alpha);// / (ndsum[m] + K * alpha);
	            
	            
	        }
        }else
        {
        	 for (int k = 0; k < K; k++) {
                 p[k] = (nw[pseData.get(m).get(n)][k] + beta) / (nwsum[k] + V * beta)
                     * (nd[m][k] + alpha) * Math.exp(neiProb[k]);// / (ndsum[m] + K * alpha);
                 
                 
             }
        }
        // cumulate multinomial parameters
        for (int k = 1; k < p.length; k++) {
            p[k] += p[k - 1];
        }
        // scaled sample because of unnormalised p[]
        double u = Math.random() * p[K - 1];
        for (topic = 0; topic < p.length; topic++) {
            if (u < p[topic])
                break;
        }

        // add newly estimated z_i to count variables
        nw[pseData.get(m).get(n)][topic]++;
        nd[m][topic]++;
        nwsum[topic]++;
       // ndsum[m]++;

        return topic;
    }
    
    double [] computNeiProb(int m, int n)
    {
    	double []neiProb = new double[K];
    	
    	ArrayList<Integer> neiIndArr = sentNeighArr.get(m).getNei(pseData.get(m).get(n));
    	
    	if(neiIndArr==null)
    		return null;
    	for(int i=0; i<neiIndArr.size(); i++)
			neiProb[z[m][neiIndArr.get(i)]]++;
    	
    	for(int i=0; i<K; i++)
    		neiProb[i] /= neiIndArr.size();
    	
    	return neiProb;
    }
    /**
     * Add to the statistics the values of theta and phi for the current state.
     *//*
    private void updateParams() {
        for (int m = 0; m < sData.size(); m++) {
            for (int k = 0; k < K; k++) {
                thetasum[m][k] += (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
            }
        }
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                phisum[k][w] += (nw[w][k] + beta) / (nwsum[k] + V * beta);
            }
        }
        numstats++;
    }*/
    
    private void gibbs(int K, double alpha, double beta) {
        this.K = K;
        this.alpha = alpha;
        this.beta = beta;

        // init sampler statistics
        if (SAMPLE_LAG > 0) {
            thetasum = new double[pseData.size()][K];
            phisum = new double[K][V];
            numstats = 0;
        }

        // initial state of the Markov chain:
        initialState(K);

        System.out.println("Sampling " + ITERATIONS
            + " iterations with burn-in of " + BURN_IN + " (B/S="
            + THIN_INTERVAL + ").");

        for (int i = 0; i < ITERATIONS; i++) {

            // for all z_i
            for (int m = 0; m < z.length; m++) {
                for (int n = 0; n < z[m].length; n++) {

                    // (z_i = z[m][n])
                    // sample from p(z_i|z_-i, w)
                	
                	
                	
                    int topic = sampleFullConditional(m, n);
                    z[m][n] = topic;
                }
            }

            if ((i < BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("B");
                dispcol++;
            }
            // display progress
            if ((i > BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("S");
                dispcol++;
            }
            // get statistics after burn-in
            if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
               updateParams();
                System.out.print("|");
                if (i % THIN_INTERVAL != 0)
                    dispcol++;
            }
            if (dispcol >= 100) {
                System.out.println();
                dispcol = 0;
            }
        }
    }
    
    /**
     * Add to the statistics the values of theta and phi for the current state.
     */
    private void updateParams() {
        for (int m = 0; m < pseData.size(); m++) {
            for (int k = 0; k < K; k++) {
                thetasum[m][k] += (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
            }
        }
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                phisum[k][w] += (nw[w][k] + beta) / (nwsum[k] + V * beta);
            }
        }
        numstats++;
    }
    
   private double[] computClusterDist(int textInd)
   {
	   double clusterDist[] = new double[numPse];
	   
	   for(int i=0; i<textDist[textInd].length; i++)
	   {
		   if(i==textInd)
			   continue;
		   
		   clusterDist[ps[i]] += textDist[textInd][i];
	   }
	   
	   for(int i=0; i<clusterDist.length; i++)
		   clusterDist[i] /= m_ps[i];
	   
	   return clusterDist;
   }
   
   private int clusterBasedDist(int m)
   {
	   m_ps[ps[m]]--;
	   double []clusterDist = computClusterDist(m);
	   
	   int ind = -1;
	   double minDist = 1000;
	   
	   for(int i=0; i<clusterDist.length; i++)
	   {
		   if(clusterDist[i]<minDist)
		   {
			   minDist = clusterDist[i];
			   ind = i;
		   }
	   }
	   m_ps[ind]++;
	   return ind;
   }
   
    private void kmeans(int pseDoc) 
    {
    	numPse = pseDoc;
        System.out.println("kmeans");
        
        ps = new int[sData.size()];
        m_ps = new int[numPse];
        for(int i=0; i<sData.size(); i++)
        {
        	int topic = (int) (Math.random() * pseDoc);
       	 	ps[i] = topic;
       	 	m_ps[topic]++;
        }
        
        for(int i=0; i<200; i++)
        {
        	
        	for (int m = 0; m < sData.size(); m++)
        	{
        		//System.out.println("m : " + m);
        		 //int topic = sampleFullConditional(m);
        		int topic = clusterBasedDist(m);
                 ps[m] = topic;
        	}
        }
        
       /* for(int i=0; i<z.length; i++)
        {
        	//System.out.print( z[i] + " ");
        //	if((i+1)%20 ==0)
        		//System.out.println();
        	cluster_doc[z[i]][i] = 1;
        }*/
    }
      
    public void test()
    {
    	System.out.println("the number of cluster " + clustering);
    
    }
    
    public void readVector()
	{
		try
		{
			BufferedReader br1 = new BufferedReader(new FileReader(vectorPath));
			String line = "";
			float vector = 0;
			while ((line = br1.readLine()) != null) 
			{
			
				String word[] = line.split(" ");
				
				String word1 = word[0];
				float []vec = new float[word.length-1];
				for(int i=1; i<word.length; i++)
				{
					vector = Float.parseFloat(word[i]);///(word.length-1);
					vec[i-1] = vector;
				}
				wordMap.put(word1, vec);
			}
			br1.close();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
    public void readText( String path)
	 {
	    	String csvFile = path;
	    	BufferedReader br = null;
			String line = "";
			String cvsSplitBy = " ";
			sData = new ArrayList<ArrayList<Integer>>();
			try {
		 
				br = new BufferedReader(new FileReader(csvFile));
			
				HashSet<Integer> allSet = new HashSet<Integer>();
				while ((line = br.readLine()) != null) 
				{
		 
					String[] num = line.split(cvsSplitBy);
						
					ArrayList<Integer> sample = new ArrayList<Integer>();
			
					for(int i=0; i<num.length; i++)
					{
						if(num[i].equals(""))
							continue;
						int word = Integer.parseInt(num[i]);
						
						if(!allSet.contains(word))
							allSet.add(word);
						sample.add(word);
						
					}
					sData.add(sample);
					
				}
			
				System.out.println("the number of differn words in data: " + allSet.size());
				br.close();
		 
			} catch(Exception e)
			{
				e.printStackTrace();
			}
	    }
    
    public double computNMI()
    {
    	//double res = 0;
    	
    	ArrayList<ArrayList<Integer>> textLabel = new ArrayList<ArrayList<Integer>>();
    	
    	ArrayList<Integer> labelId = new ArrayList<Integer>();
    	
    	for(int i=0; i< lablesArr.size(); i++)
    	{
    		int id = lablesArr.get(i);
    		
    		if(labelId.contains(id))
    		{
    			int index = labelId.indexOf(id);
    			
    			textLabel.get(index).add(i);
    		}else
    		{
    			ArrayList<Integer> subLabel = new ArrayList<Integer>();
    			subLabel.add(i);
    			labelId.add(id);
    			textLabel.add(subLabel);
    		}
    	}
    	
    	ArrayList<ArrayList<Integer>> clusterLabel = new ArrayList<ArrayList<Integer>>();
    	
    	ArrayList<Integer> clusterlId = new ArrayList<Integer>();
    	
    	for(int i=0; i<ps.length ; i++)
    	{
    		int id = ps[i];
    		
    		if(clusterlId.contains(id))
    		{
    			int index = clusterlId.indexOf(id);
    			
    			clusterLabel.get(index).add(i);
    		}else
    		{
    			ArrayList<Integer> subLabel = new ArrayList<Integer>();
    			subLabel.add(i);
    			clusterlId.add(id);
    			clusterLabel.add(subLabel);
    		}
    	}
    	
    	System.out.println(" the cluster number : " + clusterLabel.size());
    	double comRes = 0;
    	
    	for(int i=0; i<textLabel.size(); i++)
    	{
    		for(int j=0; j<clusterLabel.size(); j++)
    		{
    			int common = commonArray(textLabel.get(i),clusterLabel.get(j));
    			
    			if(common!=0)
    				comRes += (double)common*Math.log((double)ps.length*common/(textLabel.get(i).size()*clusterLabel.get(j).size()));
    		}	
    	}
    	
    	double comL = 0;
    	for(int i=0; i<textLabel.size(); i++)
    	{
    		comL += (double)textLabel.get(i).size()*Math.log((double)textLabel.get(i).size()/ps.length);
    	}
    	
    	double comC = 0;
    	for(int j=0; j<clusterLabel.size(); j++)
    		comC += (double)clusterLabel.get(j).size()*Math.log((double)clusterLabel.get(j).size()/ps.length);
    	
    	//System.out.println(comRes + " " + comL + " "+ comC);
    	
    	comRes /= Math.sqrt(comL*comC);
    	/*for(int i=0; i<clusterLabel.size(); i++)
    	{
    		System.out.println(i + " " +clusterLabel.get(i).toString());
    	}*/
    	
    	return comRes;
    }
    
    public int commonArray(ArrayList<Integer> arr1, ArrayList<Integer> arr2)
    {
    	int count = 0;
    	for(int i=0; i<arr1.size(); i++)
    		if(arr2.contains(arr1.get(i)))
    			count++;
    	
    	return count;
    }
    
    public void readLable( String path)
    {
    	
    	BufferedReader br = null;
 		String line = "";
 		String cvsSplitBy = " ";
 		
 		try {
 	 
 			br = new BufferedReader(new FileReader(path));

 			//ArrayList<Integer> vArr = new ArrayList<Integer>();
 			
 			while ((line = br.readLine()) != null) {
 	 
 			        // use comma as separator
 				String[] num = line.split(cvsSplitBy);
 				
 				//System.out.println(num[0]);
 				//int laberNum = Integer.parseInt(num[0]);
 				
 				//label.add(laberNum);
 				//wordId.put(Integer.parseInt(num[0]), num[1]);
 				lablesArr.add(Integer.parseInt(num[0]));
 			}
 			//V = wordsArr.size();
 	 
 		} catch (FileNotFoundException e) {
 			e.printStackTrace();
 		} catch (IOException e) {
 			e.printStackTrace();
 		} finally {
 			if (br != null) {
 				try {
 					br.close();
 				} catch (IOException e) {
 					e.printStackTrace();
 				}
 			}
 		}
    }
    
    
    public void readWord( String path)
    {
    	
    	BufferedReader br = null;
		String line = "";
		String cvsSplitBy = " ";
		
		try {
	 
			br = new BufferedReader(new FileReader(path));

			//ArrayList<Integer> vArr = new ArrayList<Integer>();
			
			while ((line = br.readLine()) != null) {
	 
			        // use comma as separator
				String[] num = line.split(cvsSplitBy);
				
				//System.out.println(num[0]);
				//int laberNum = Integer.parseInt(num[0]);
				
				//label.add(laberNum);
				//wordId.put(Integer.parseInt(num[0]), num[1]);
				wordsArr.add(num[0]);
			}
			
			V = wordsArr.size();
			System.out.println(" the number of words in wordList: " + wordsArr.size());
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	 
		
    }
    
    
    public ArrayList<Double> normBagOfWords(ArrayList<Integer> doc, ArrayList<Integer> diff)
	{
		ArrayList<Double> freArr = new ArrayList<Double>();
		
		HashMap<Integer,Integer> wordF = new HashMap<Integer,Integer>();
		
		for(int i=0; i<doc.size(); i++)
		{
			if(!wordMap.containsKey(wordsArr.get(doc.get(i))))
				continue;
			if(!wordF.containsKey(doc.get(i)))
			{
				wordF.put(doc.get(i), 1);
				diff.add(doc.get(i));
			}
			else
				wordF.put(doc.get(i), wordF.get(doc.get(i))+1);
		}
		
		for(int i=0; i<diff.size(); i++)
		{
			freArr.add((double)wordF.get(diff.get(i))/doc.size());
		}
		
		return freArr;
	}
    
    public void configure(int iterations, int burnIn, int thinInterval,
            int sampleLag) {
            ITERATIONS = iterations;
            BURN_IN = burnIn;
            THIN_INTERVAL = thinInterval;
            SAMPLE_LAG = sampleLag;
        }
    
    
    public double docDist(ArrayList<Double> d1, ArrayList<Double> d2, ArrayList<Double> dist)
	{
		double sis = 0.0;
		
		ArrayList<Double> doc1 = new ArrayList<Double> ();
		doc1.addAll(d1);
		
		ArrayList<Double> doc2 = new ArrayList<Double> ();
		doc2.addAll(d2);
		
		ArrayList<Integer> indArr = new ArrayList<Integer>();
		
		
		for(int i=0; i<dist.size(); i++)
		{
			indArr.add(i);
		}
		
		for(int i=0; i<dist.size(); i++)
		{
			double cur = dist.get(i);
			
			int curInd = indArr.get(i);
			
			double minV = dist.get(i);
			int minInd = i;
			int ind = -1;
			for(int j=i+1; j<dist.size(); j++)
			{
				if(dist.get(j)<minV)
				{
					minV = dist.get(j);
					minInd = j;
					ind = indArr.get(j);
				}
			}
			if(i!=minInd)
			{
				
				dist.set(i, minV);
				dist.set(minInd, cur);
				
				indArr.set(i, ind);
				indArr.set(minInd, curInd);
			
			}
			//indArr.add(minInd);
		}
		
		for(int i=0; i<dist.size(); i++)
		{
			int index = indArr.get(i);
			
			int doc1Ind = index/doc2.size();
			int doc2Ind = index%doc2.size();
			
			double wei1 = doc1.get(doc1Ind);
			double wei2 = doc2.get(doc2Ind);
			
			if(wei1<1e-5 || wei2<1e-5)
				continue;
			
			double minWei = 0.0;
			if(wei1>wei2)
				minWei = wei2;
			else
				minWei = wei1;
			
			sis += minWei*dist.get(i);
			doc1.set(doc1Ind, wei1-minWei);
			doc2.set(doc2Ind, wei2-minWei);	
		}
		
		return sis;
	}
	
    public double docDist2(ArrayList<Double> d1, ArrayList<Double> d2, ArrayList<Double> distArr)
	{
		double dist = 0.0;
		
		int ind = 1;
		
		double minDist = 1000;
		
		for(int i=0; i<distArr.size(); i++)
		{
			if(distArr.get(i)<minDist)
				minDist = distArr.get(i);
			if((i+1)/d2.size()==ind)
			{
				dist += d1.get(ind-1)*minDist;
				ind++;
				minDist = 1000;
			}
		}
		return dist;
	}
    
   
    public void computWordDist()
    {
    	System.out.println("the function of wordDist");
    	wordDist = new double[wordsArr.size()][wordsArr.size()];
    	
    	for(int i=0; i<wordDist.length; i++)
    		for(int j=0; j<wordDist[i].length; j++)
    			wordDist[i][j] = 1;
    	
    	for(int i=0; i<wordDist.length; i++)
    		wordDist[i][i] = 0;
    	
    	int noInclude = 0;
    	for(int i=0; i<wordDist.length-1; i++)
    	{
    		String word = wordsArr.get(i);
    		
    		if(!wordMap.containsKey(word))
    		{
    			//System.out.println(word);
    			//for(int j=0; j<wordDist.length)
    			noInclude++;
				continue;
    		}
    		
    		float [] wordVect1 = wordMap.get(word);
    		for(int j= i+1; j<wordDist[i].length; j++)
    		{
    			String word2 = wordsArr.get(j);
        		
        		if(!wordMap.containsKey(word2))
    				continue;
        		
        		float [] wordVect2 = wordMap.get(word2);
        		double dist = (double)(TermVector.computDist(wordVect1, wordVect2));
        		//float dist = TermVector.computDist(wordVect1,wordVect2);
        		//if(dist<0 || dist>=1)
        			//System.out.println("Wrong" + dist);
        		if(dist>1)
        			dist = 1;
        		if(dist<0 || dist>1)
        			System.out.println("Wrong" + dist);
    			wordDist[i][j] = dist;
    			wordDist[j][i] = dist;
    			
    		}
    	}
    	
    	System.out.println("the number of words isnot in word2vec: " + noInclude);
    	
    }
    
    
    public ArrayList<Double> textWordDist(ArrayList<Integer> d1, ArrayList<Integer> d2)
    {
    	ArrayList<Double> dist = new ArrayList<Double>();
    	
    	for(int i=0; i<d1.size(); i++)
    	{
    		//float [] wordVect1 = wordMap.get(wordsArr.get(d1.get(i)));
    		
    		for(int j=0; j<d2.size(); j++)
    		{
    			//float [] wordVect2 = wordMap.get(wordsArr.get(d2.get(j)));
    			
    			//float wordDist = TermVector.computDist(wordVect1,wordVect2);
    			dist.add(wordDist[d1.get(i)][d2.get(j)]);
				
    		}
    	}
    	
    	return dist;
    }
    
    public double compuDistIndex(int ind1, int ind2)
    {
    	ArrayList<Integer> uniqueText1 = new ArrayList<Integer>();
		
		ArrayList<Double> weightText1 = normBagOfWords(sData.get(ind1),uniqueText1);
		
		ArrayList<Integer> uniqueText2 = new ArrayList<Integer>();
		
		ArrayList<Double> weightText2 = normBagOfWords(sData.get(ind2),uniqueText2);
		
		ArrayList<Double> dist = textWordDist(uniqueText1, uniqueText2);
		
		double twoTextDist = docDist2(weightText1,weightText2,dist);
		return twoTextDist;
		
    }
    
    public void testTestDist()
    {
    	int ind1 = 0;
    	int ind2 = 1;
    	int ind3 = sData.size()-1;
    	int ind4 = sData.size()-2;
    	System.out.println(compuDistIndex(ind1,ind2));
    	System.out.println(compuDistIndex(ind1,ind3));
    	System.out.println(compuDistIndex(ind1,ind4));
    	//ystem.out.println(compuDistIndex(ind2,ind2));
    	
    }
    
    public void getMinIndex(double []array, int cur, BufferedWriter bw) throws Exception
    {
    	ArrayList<Integer> nei = new ArrayList<Integer>();

       for(int i = 0; i < array.length; i++) {
        	if(i==cur)
        		continue;
        	
        	if(array[i]<=0.4)
        	{
        		bw.write(String.valueOf(i) + " ");
        	}
          
        }
       bw.newLine();
       // retur
    }
       
    public void computWordNei() throws Exception 
    {
    	//wordNei = new int[wordDist.length][3];
    	
    	//for(int i=0; i<wordNei.)
    	
    	String neiPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName +"/wordNeighbors0.4.txt";;
    	FileWriter fw = new FileWriter(neiPath);
		BufferedWriter bw = new BufferedWriter(fw);
		
		
    	for(int i=0; i<wordDist.length; i++)
    	{
    		//wordNei[i] = getMinIndex(wordDist[i],3,i);
    		getMinIndex(wordDist[i], i, bw);
    	//	System.out.println(wordsArr.get(i) + " " + wordsArr.get(wordNei[i][0])+ " " + wordsArr.get(wordNei[i][1])
    	//	+ " " + wordsArr.get(wordNei[i][2])+ " " + wordsArr.get(wordNei[i][3])+ " " + wordsArr.get(wordNei[i][4]));
    	}
    	bw.close();
    	fw.close();
    }
    
    
    public void computDistForTwoText() throws Exception
    {
    	computWordDist();
    	
    	computWordNei();
    	
    	System.out.println("computDistForTwoText");
    	textDist = new double[sData.size()][sData.size()];
    	
    	for(int i=0; i<sData.size()-1; i++)
    	{
    		System.out.println(i);
    		ArrayList<Integer> uniqueText1 = new ArrayList<Integer>();
    		
    		ArrayList<Double> weightText1 = normBagOfWords(sData.get(i),uniqueText1);
    		//boolean flag = false;
    		//int cur = -1;
    		for(int j=i+1; j<sData.size(); j++)
    		{
    			/*if((j%2)==0)
    			{
    				cur = j;
    				flag = true;
    				
    				j = sData.size()-j;
    				
    				System.out.println("reverse:" + j);
    			}
    			else
    				System.out.println("right:" + j);*/
    			ArrayList<Integer> uniqueText2 = new ArrayList<Integer>();
        		
        		ArrayList<Double> weightText2 = normBagOfWords(sData.get(j),uniqueText2);
        		
        		ArrayList<Double> dist = textWordDist(uniqueText1, uniqueText2);
        		
        		double twoTextDist = docDist2(weightText1,weightText2,dist);
        		textDist[i][j] = twoTextDist;
        		textDist[j][i] = twoTextDist;
        		//System.out.println(twoTextDist);
        		/*if(flag)
        		{
        			flag = false;
        			j = cur;
        		}*/
    		}
    	}
    	
    }
    
    public double[][] getPhi() {
        double[][] phi = new double[K][V];
        if (SAMPLE_LAG > 0) {
        	System.out.println("getPhi");
            for (int k = 0; k < K; k++) {
                for (int w = 0; w < V; w++) {
                    phi[k][w] = phisum[k][w] / numstats;
                }
            }
        } else {
            for (int k = 0; k < K; k++) {
                for (int w = 0; w < V; w++) {
                    phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
                }
            }
        }
        return phi;
    }
    
    public double[][] getTheta() {
        double[][] theta = new double[pseData.size()][K];

        if (SAMPLE_LAG > 0) {
        	for (int m = 0; m < pseData.size(); m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = thetasum[m][k] / numstats;
                }
            }
        } 
        else{
            for (int m = 0; m < pseData.size(); m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
                }
            }
        }
        return theta;
    }

    
    public void getCluster()
    {
    	//clu = new int[sData.size()];
    	
    	for (int m = 0; m < pseData.size(); m++)
    	{
    		ArrayList<Integer> shortInd = shortIndArr.get(m);
    		ArrayList<Integer> longInd = longIndArr.get(m);
    		
    		//ArrayList<Integer> pseD = pseData.get(m);
    		
    		int preInd = 0;
    		
    		for(int i=0; i<shortInd.size(); i++)
    		{
    			
    			int m_k[] = new int[K];
    			for(; preInd<longInd.get(i); preInd++)
    			{
    				m_k[z[m][preInd]]++;
    			}
    			
    			int maxI = 0;
        		int val = 0;
        		for (int k = 0; k < K; k++) 
        		{
	                  if(m_k[k]>val)
	                  {
	                      maxI = k;
	                      val = m_k[k];
	                   }
                 }
        		ps[shortInd.get(i)] = maxI;    			
    			preInd = longInd.get(i);
    		}
    	}
    	
    }
    
    public int maxIndexArr(double []arr)
    {
    	int maxI = 0;
		double val = 0;
		for (int k = 0; k < arr.length; k++) 
		{
              if(arr[k]>val)
              {
                  maxI = k;
                  val = arr[k];
               }
         }
		return maxI;
    }
    public void getCluster3()
    {
    	//clu = new int[sData.size()];
    	double theta[][] = getTheta();
    	
    	for (int m = 0; m < theta.length; m++)
    	{
    		ArrayList<Integer> shortInd = shortIndArr.get(m);
    		//ArrayList<Integer> longInd = longIndArr.get(m);
    		
    		int maxK = maxIndexArr(theta[m]);
    		
    		for(int i=0; i<shortInd.size(); i++)
    		{
        		ps[shortInd.get(i)] = maxK;    
    		}
    	}
    	
    }
    
    public void getClusterDMM()
    {
    	//clu = new int[sData.size()];
    	for(int i=0; i<ps.length; i++)
    		ps[i] = -1;
    	
    	int topic;
    	for (int m = 0; m < pseData.size(); m++)
    	{
    		ArrayList<Integer> shortInd = shortIndArr.get(m);
    		ArrayList<Integer> longInd = longIndArr.get(m);
    		
    		ArrayList<Integer> pseD = pseData.get(m);
    		
    		int preInd = 0;
    		
    		for(int i=0; i<shortInd.size(); i++)
    		{
    			int len = longInd.get(i)-preInd;
    			double[] p = new double[K];
    			
    			int m_k[] = new int[K];
    		    for (int k = 0; k < K; k++) 
    		    {
    		        	//System.out.println("K: " + k);
    		        double wordsS = 1;
    		        int begin = preInd;
    		    	for(; begin<longInd.get(i); begin++)
    		    	{
    		    		m_k[z[m][begin]]++;
    		    		wordsS *= (nw[pseD.get(begin)][k] + beta);
    		    	}
    		    			
    		    		
    		    	double wordsT = 1;
    		    	for(int j=1; j<=len; j++)
    		    		wordsT *= (nwsum[k] + V*beta);
    		    		
    		        //p[k] = (nd[m][k] + alpha)* wordsS  / (wordsT);
    		    	p[k] =  wordsS  / wordsT;
    		     }
    		
    		        // cumulate multinomial parameters
    		     for (int k = 1; k < p.length; k++) 
    		            p[k] += p[k - 1];
    		            //System.out.println(p[k]);
    		        // scaled sample because of unnormalised p[]
    		        
    		     double u = Math.random() * p[K - 1];
    		       // System.out.println("u: " + u);
    		       // System.out.println("p : " + p[p.length-1]);
    		        for (topic = 0; topic < p.length; topic++) {
    		            if (u < p[topic])
    		                break;
    		        }

        		ps[shortInd.get(i)] = topic;    	
        		
        		if(topic==K)
        		{
        			int maxI = 0;
            		int val = 0;
            		for (int k = 0; k < K; k++) 
            		{
    	                  if(m_k[k]>val)
    	                  {
    	                      maxI = k;
    	                      val = m_k[k];
    	                   }
                     }
            		ps[shortInd.get(i)] = maxI;   
            		System.out.println("getClusterDMM"  + " " + maxI);
        		}
        			
    			preInd = longInd.get(i);
    		}
    	}
    	
    }
  
    
    public void getCluster2()
    {
    	//clu = new int[sData.size()];
    	
    	double phi[][] = getPhi();
    	
    	for(int i=0; i<K; i++)
    		System.out.print(phi[i][0] + " ");
    	
    	int topic;
    	for (int m = 0; m < pseData.size(); m++)
    	{
    		ArrayList<Integer> shortInd = shortIndArr.get(m);
    		ArrayList<Integer> longInd = longIndArr.get(m);
    		
    		ArrayList<Integer> pseD = pseData.get(m);
    		
    		int preInd = 0;
    		
    		for(int i=0; i<shortInd.size(); i++)
    		{
    			//int len = longInd.get(i)-preInd;
    			double[] p = new double[K];
    		    for (int k = 0; k < K; k++) 
    		    {
    		        	//System.out.println("K: " + k);
    		        //double wordsS = 1;
    		        int begin = preInd;
    		    	for(; begin<longInd.get(i); begin++)
    		    			p[k] *= phi[k][pseD.get(begin)];//(nw[][k] + beta);
    		     }
    		
    		        // cumulate multinomial parameters
    		     for (int k = 1; k < p.length; k++) 
    		            p[k] += p[k - 1];
    		            //System.out.println(p[k]);
    		        // scaled sample because of unnormalised p[]
    		        
    		     double u = Math.random() * p[K - 1];
    		       // System.out.println("u: " + u);
    		       // System.out.println("p : " + p[p.length-1]);
    		        for (topic = 0; topic < p.length; topic++) {
    		            if (u < p[topic])
    		                break;
    		        }

        		ps[shortInd.get(i)] = topic;    
        		System.out.println(shortInd.get(i) + " " + topic);
    			preInd = longInd.get(i);
    		}
    	}
    	
    }
  
    public int[]  getMaxIndex(double []array, int top_k)
    {
    	double[] max = new double[top_k];
        int[] maxIndex = new int[top_k];
        Arrays.fill(max, Double.NEGATIVE_INFINITY);
        Arrays.fill(maxIndex, -1);

        top: for(int i = 0; i < array.length; i++) {
            for(int j = 0; j < top_k; j++) {
                if(array[i] > max[j]) {
                    for(int x = top_k - 1; x > j; x--) {
                        maxIndex[x] = maxIndex[x-1]; max[x] = max[x-1];
                    }
                    maxIndex[j] = i; max[j] = array[i];
                    continue top;
                }
            }
        }
        return maxIndex;

    }
    
    public double mainFun(int times, int topics, int topWords) throws Exception
    {
    
    	 int K = topics;
          
    	 //int M = sData.size();
    	 String resPath = "";
         if(times>0)
       	  resPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName + "/TopWords/STTP_topics=" + topics + "topWords="+topWords+"times="+times+".txt";
        	// resPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/s20NewsRes/ShortResult/MergeShort_topics=" + topics + "topWords="+topWords+"times="+times+".txt";
         else
       	  resPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName + "/TopWords/STTP_topics=" + topics + "topWords="+topWords+".txt";

          FileWriter fw = new FileWriter(resPath);
  		  BufferedWriter bw = new BufferedWriter(fw);
  		  
  		  double alpha1 = 0.1;
		  
		  double beta1 = 0.1;
          
           gibbs(K, alpha1, beta1);
         
          getClusterDMM();
          double nmi = computNMI();
          System.out.println(nmi);
          
          double[][] phi = getPhi();
          
          for(int i=0; i<phi.length; i++)
          {
       	 //  ArrayList<String> wArr = new ArrayList<String>();
       	   
       	   	   int[] maxIndices = getMaxIndex(phi[i],topWords);
              
	       	   for(int j=0; j<maxIndices.length; j++)
	       	   {
	       		  // wArr.add(wordId.get(maxIndices[j]));
	       		   int ind = maxIndices[j];
	       		   bw.write(wordsArr.get(ind));
	       		   bw.write(" ");
	       	   }
       	       bw.newLine();
       	    bw.newLine();
          }
        // bw.write(res);
         bw.close();
         fw.close();
         return nmi;
    }
    
    public void load_wordid_neighcnt(String neiPath)
	 {
		 BufferedReader br = null;
			String line = "";
			String cvsSplitBy = " ";
			
			try {
		 
				br = new BufferedReader(new FileReader(neiPath));

				//ArrayList<Integer> vArr = new ArrayList<Integer>();
				
				int wordInd = 0;
				while ((line = br.readLine()) != null) {
					String[] num = line.split(cvsSplitBy);
					
					if(num[0].equals(""))
						num_neighbors_foreach_word[wordInd] = 0;
					else
					{
						num_neighbors_foreach_word[wordInd] = num.length;
						word_neighbors[wordInd] = new int[num.length];
						
						for(int i=0; i<num.length; i++)
						{
							word_neighbors[wordInd][i] = Integer.parseInt(num[i]);
						}
					}
					wordInd++;
				}
				
		 
			} catch (Exception e) {
				e.printStackTrace();
			}
	 }
	
    public void computNeiIndex()
    {
    	sentNeighArr = new ArrayList<NeightNode>();
    	for(int i=0; i<pseData.size(); i++)
    	{
    		NeightNode nn = new NeightNode();
    		
    		HashSet<Integer> idSet = new HashSet<Integer>();
    		for(int j=0; j<pseData.get(i).size(); j++)
    		{
    			int id = pseData.get(i).get(j);
    			if(idSet.contains(id))
    			{
    				continue;
    			}else
    			{
    				idSet.add(id);
    				
    				int neiNum = num_neighbors_foreach_word[id];
    				
    				if(neiNum<1)
    					continue;
    				
    				for(int n=0; n<neiNum; n++)
    				{
    					int neiId = word_neighbors[id][n];
    					
    					for(int ind=0; ind<pseData.get(i).size(); ind++)
        				{
    						if(pseData.get(i).get(ind)==neiId)
    							nn.add(id, ind);
          				}
    				}
    				
    			}
    		}
    		sentNeighArr.add(nn);
    	}
    }
    
    
    public void combineShortTexts()
    {
    	System.out.println("combine short texts into pse texts");
    	pseData = new ArrayList<ArrayList<Integer>>();
    	shortIndArr = new ArrayList<ArrayList<Integer>>();
    	longIndArr = new ArrayList<ArrayList<Integer>>();
    	
    	for(int i=0; i<numPse; i++)
    	{
    		ArrayList<Integer> pseText = new ArrayList<Integer>();
    		ArrayList<Integer> shortArr = new ArrayList<Integer>();
    		ArrayList<Integer> longArr = new ArrayList<Integer>();
    		
    		int longPse = 0;
    		for(int j=0; j<ps.length; j++)
        	{
        		if(ps[j]==i)
        		{
        			shortArr.add(j);
        			for(int w=0; w<sData.get(j).size(); w++)
        			{
        				int wInd = sData.get(j).get(w);
        				//int wNum = sDataNum.get(j).get(w);
        				
        				//for(int n=0; n<wNum; n++)
        					pseText.add(wInd);
        				
        				
        			}
        			longPse += sData.get(j).size();
        			longArr.add(longPse);
        		}
        	}
    		if(shortArr.size()>0)
    		{
    			pseData.add(pseText);
    			shortIndArr.add(shortArr);
    			longIndArr.add(longArr);
    		}
    	}
    }
    public void printSis() throws Exception
    {
    	String sisPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/googlenews/textSis.txt";
    	//readLable("C:/Users/jipeng/Desktop/TopicModel/dataset/s20NewsRes/ShortText/sentenceLabel.txt");
    	
    	//String nmiPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/s20NewsRes/EmbKmeans_NMI.txt";
    	FileWriter fw = new FileWriter(sisPath);
		BufferedWriter bw = new BufferedWriter(fw);
		
		DecimalFormat myFormatter = new DecimalFormat("0.000");
		
		for(int i=0; i<textDist.length; i++)
		{
			for(int j=0; j<textDist[i].length; j++)
			{
				bw.write(String.valueOf(myFormatter.format(1-textDist[i][j])) + " ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();
    }
    
    public void iterMain(String path, String wordPath) throws Exception
    {
    	readText(path);
    	readWord(wordPath);
    	
    	readVector();
    	//testTestDist();
    	computDistForTwoText();
    	
    	String neiPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName +"/wordNeighbors0.4.txt";
    	
    	num_neighbors_foreach_word=new int[V];
    	//allocate neighbors for each word
    	word_neighbors=new int[V][];
    	load_wordid_neighcnt(neiPath);
    	
    	
    	//printSis();
    	
    	readLable("C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName + "/label.txt");
    	
    	String nmiPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dataName + "/NMI/STTP2.txt";
    	//readLable("C:/Users/jipeng/Desktop/TopicModel/dataset/s20NewsRes/ShortText/sentenceLabel.txt");
    	
    	//String nmiPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/s20NewsRes/MergeShort_NMI.txt";
    	FileWriter fw = new FileWriter(nmiPath);
		BufferedWriter bw = new BufferedWriter(fw);
		
        System.out.println("ETM using Gibbs Sampling.");

        // LdaGibbsSampler lda = new LdaGibbsSampler(documents, V);
        // configure(1000, 500, 100, 5);
        int pseDocNum = sData.size()/50;
        kmeans(pseDocNum);
        
        double nmi = computNMI();
        System.out.println("Kmeans:" + nmi);
        combineShortTexts();
        
        computNeiIndex();
        configure(1000, 500, 20, 5);
        
        int topics = 10;
        
        for(int times=1; times<=5; times++ )
       // {
        	for(int k=topics; k<=10; k=k+20)
         	{
         		for(int topWords=10; topWords<=10; topWords=topWords+20)
        		{
        			
        			nmi = mainFun(times, k, topWords);
        			
        			// double nmi = computNMI();
        	          System.out.println(nmi);
             		bw.write(String.valueOf(nmi) + " ");
             		bw.newLine();
        		}
        	}
        	
        	bw.close();
            fw.close(); 
        //}
    }
    
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		
       //String dataName = "Tweet";
		//String dataName = "NIPS";
        String dataName = "s_googlenews";
        
		ETM dmm = new ETM(dataName);
      
        try
		{
			  
        	String path = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dmm.dataName +"/sentences.txt";
			 String wordPath = "C:/Users/jipeng/Desktop/TopicModel/dataset/" + dmm.dataName +"/word.txt";
			
			  dmm.iterMain(path, wordPath);
		}catch(Exception e)
		{
			e.printStackTrace();
		}
        
        
       
	}

}


