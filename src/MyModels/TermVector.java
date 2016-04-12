package MyModels;


public class TermVector
    {
        public static float ComputeCosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1.length != vector2.length)
				try {
					throw new Exception("DIFER LENGTH");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}


            float denom = (VectorLength(vector1) * VectorLength(vector2));
            if (denom == 0f)
                return 0f;
            else
                return (InnerProduct(vector1, vector2) / denom);

        }
        
        public static float computDist(float[] vector1, float[] vector2)
        {
           
               // return (float)Math.exp((1-ComputeCosineSimilarity(vector1,vector2))/0.3f);
        	return 1-ComputeCosineSimilarity(vector1,vector2);

        }


        public static float InnerProduct(float[] vector1, float[] vector2)
        {

            if (vector1.length != vector2.length)
				try {
					throw new Exception("DIFFER LENGTH ARE NOT ALLOWED");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}


            float result = 0f;
            for (int i = 0; i < vector1.length; i++)
                result += vector1[i] * vector2[i];

            return result;
        }

        public static float VectorLength(float[] vector)
        {
            float sum = 0.0f;
            for (int i = 0; i < vector.length; i++)
                sum = sum + (vector[i] * vector[i]);

            return (float)Math.sqrt(sum);
        }

  }

