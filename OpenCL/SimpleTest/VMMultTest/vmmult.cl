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

void VMMult(__global const Vec3 * a, __global const Mat4x4 * m, __local Vec3 * r)
{
	r->x = (a->x * m->_11 + a->y * m->_21 + a->z * m->_31 + m->_41);
	r->y = (a->x * m->_12 + a->y * m->_22 + a->z * m->_32 + m->_42);
	r->z = (a->x * m->_13 + a->y * m->_23 + a->z * m->_33 + m->_43);
}

__kernel void Mult(__global const Vec3 * a, __global Mat4x4 * m, __global Vec3 * r)
{
	int gid = get_global_id(0);
	__local int i;
	__local int isz;
	int o = gid * 5000;
	__local Vec3 sumR;
	__local Vec3 ta;
	__local Vec3 tv;
	sumR.x = 0.0f;
	sumR.y = 0.0f;
	sumR.z = 0.0f;
	
	isz = o + 5000;
	for(i = o ; i < isz ; i++)
	{
		VMMult(&a[i], &m[0], &tv);
		ta.x += tv.x;
		ta.y += tv.y;
		ta.z += tv.z;
	}
	r[gid].x = sumR.x;
	r[gid].y = sumR.y;
	r[gid].z = sumR.z;
}
