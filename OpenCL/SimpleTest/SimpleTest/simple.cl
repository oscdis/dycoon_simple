__kernel void Count(const __global const int * a, __global int * ct0, __global int * ct1)
{
	int gid = get_global_id(0);
	int i;
	int isz;
	int o = gid * 5000;
	int ct0t = 0;
	int ct1t = 0;
	
	isz = o + 5000;
	for(i = o ; i <isz ; i++)
	{
		if((a[i] % 2) == 0)
		{
			ct0t++;
		}
		else
		{
			ct1t++;
		}
	}
	ct0[gid] = ct0t;
	ct1[gid] = ct1t;
}
