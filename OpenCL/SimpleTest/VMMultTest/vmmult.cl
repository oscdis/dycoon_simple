typedef struct{
	float x;
	float y;
	float z;
} Vec3;

typedef struct{
	float _11;
	float _12;
	float _13;
	float _14;

	float _21;
	float _22;
	float _23;
	float _24;

	float _31;
	float _32;
	float _33;
	float _34;

	float _41;
	float _42;
	float _43;
	float _44;

} Mat4x4;

__kernel void Mult(__global const Vec3 * a, __global Mat4x4 * m, __global Vec3 * r)
{
	int gid = get_global_id(0);
	__local int i;
	__local int isz;
	int o = gid * 5000;
	__local Vec3 sumR;
	__local Vec3 ta;
	sumR.x = 0.0f;
	sumR.y = 0.0f;
	sumR.z = 0.0f;
	
	//__local float4 m0;
	//__local float4 m1;
	//__local float4 m2;
	//__local float4 v;
	
	//m0.x = m->_11; m0.y = m->_21; m0.z = m->_31; m0.w = m->_41;
	//m1.x = m->_12; m1.y = m->_22; m1.z = m->_32; m1.w = m->_42;
	//m2.x = m->_13; m2.y = m->_23; m2.z = m->_33; m2.w = m->_43;
	
	isz = o + 5000;
	for(i = o ; i <isz ; i++)
	//for(i = o ; i <isz ; i += 4)
	{
		//v.x = a[i].x; v.y = a[i].y; v.z = a[i].z; v.w = 1.0f;
		
		//sumR.x += dot(v, m0);
		//sumR.y += dot(v, m1);
		//sumR.z += dot(v, m2);
		
		//ta.x = a[i].x;
		//ta.y = a[i].y;
		//ta.z = a[i].z;
		
		//sumR.x += ta.x * m->_11 + ta.y * m->_21 + ta.z * m->_31 + m->_41;
		//sumR.y += ta.x * m->_12 + ta.y * m->_22 + ta.z * m->_32 + m->_42;
		//sumR.z += ta.x * m->_13 + ta.y * m->_23 + ta.z * m->_33 + m->_43;
		
		sumR.x += a[i].x * m->_11 + a[i].y * m->_21 + a[i].z * m->_31 + m->_41;
		sumR.y += a[i].x * m->_12 + a[i].y * m->_22 + a[i].z * m->_32 + m->_42;
		sumR.z += a[i].x * m->_13 + a[i].y * m->_23 + a[i].z * m->_33 + m->_43;
	}
	r[gid] = sumR;
}
