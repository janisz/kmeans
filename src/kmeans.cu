#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
# define WINDOWS_LEAN_AND_MEAN
# define NOMINMAX
# include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h
#include <timer.h> // timing functions

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check
#include <helper_cuda_gl.h> // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <limits.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.30f
#define REFRESH_DELAY 10 //ms

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

float3 *dptr = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0; // FPS count for averaging
int fpsLimit = 1; // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runProgram(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
		unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void prepareCuda(struct cudaGraphicsResource **vbo_resource);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

////////////////////////////////////////////////////////////////////////////////
#define MESH_SIZE (width*height)

const unsigned int window_width = 800;
const unsigned int window_height = 800;

const unsigned int width  = 256;
const unsigned int height = 256;

float3 *deviceObjects;
float3 *deviceClusters;
int *deviceMembership;
int *membership = NULL;
int *numberOfPointsThatChangeCluster;
#define CLUSTER_COUNT 5
#define OBJECTS_CLUSTER_CHANGE_THRESHOLD 0.1
////////////////////////////////////////////////////////////////////////////////



/*-----------------------------------------------------------------------------------------------------*/

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator/(const float3 &a, const int b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline __host__ __device__ float sqDistance(float3 pt1, float3 pt2)
{
	float3 v = pt2 - pt1;
	return (dot(v,v));
}


__global__
void findNearestClusterAndUpdateMembership(  float3 *objects,
											 float3 *clustersPositions,
											 int *membership,
											 int *delta
										 )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int objectId = y*width+x;

	__shared__ float3 clusters[CLUSTER_COUNT];
	__shared__ float3 newClusters[CLUSTER_COUNT];
	__shared__ int clusterSize[CLUSTER_COUNT];
	__shared__ int changedClusters[1];
	if (threadIdx.x < CLUSTER_COUNT)
	{
		newClusters[threadIdx.x] = make_float3(0, 0, 0);
		clusters[threadIdx.x] = clustersPositions[threadIdx.x];
		clusterSize[threadIdx.x] = 0;
	}
	changedClusters[0] = 0;
	__syncthreads();

    if (objectId < MESH_SIZE) {
    	int index = 0;
		float dist, min_dist;
		float3 position = objects[objectId];

		/* find the cluster id that has min sqDistance to object */
		min_dist = sqDistance(position, clusters[0]);

		for (int i=1; i<CLUSTER_COUNT; i++) {
			dist = sqDistance(position, clusters[i]);

			if (dist < min_dist) { /* find the min and its array index */
				min_dist = dist;
				index = i;
			}
		}

		atomicAdd(&clusterSize[index], 1);
		atomicAdd((float*)newClusters +index 				 	, position.x);
		atomicAdd((float*)newClusters + index +   sizeof(float)	, position.y);
		atomicAdd((float*)newClusters + index + 2*sizeof(float)	, position.z);
		if (membership[objectId] != index) {
			atomicAdd(changedClusters, 1);
			membership[objectId] = index;
		}
    }
    __syncthreads();

}

__global__
void calculateNewClustersPositions( float3 *objects, 	/* [numClusters] */
									float3 *clusters, 	/* [numClusters] */
									int *membership 	/* [numObjs] */
								 )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int clusterId = y*width+x;

	if (clusterId < CLUSTER_COUNT) {
    	float3 position = make_float3(0, 0, 0);
    	int objectsInCluster = 0;

		for (int i=0; i<MESH_SIZE; i++) {
			if (membership[i] == clusterId) {
				position = position + objects[i];
				objectsInCluster++;
			}
		}

		if (objectsInCluster != 0) {
			position = position / objectsInCluster;
		}

		clusters[clusterId] = position;
    }
}

void kmeans(int *membership, int *loop_iterations)
{
	static int initialized;
    int loop=0;
    if (initialized > 100) exit(0);
    if (!initialized) {
        //printf("/* pick first numClusters elements of objects[] as initial cluster centers*/\n");
    	float3 clusters[CLUSTER_COUNT];
    	for (int i=0;i<CLUSTER_COUNT;i++) clusters[i] = dptr[i];
        //printf("/* initialize membership[] */\n");
        for (int i=0; i<MESH_SIZE; i++) membership[i] = -1;
        CudaSafeCall( cudaMallocHost(&numberOfPointsThatChangeCluster, sizeof(int)));
    	CudaSafeCall( cudaMalloc(&deviceClusters, CLUSTER_COUNT*sizeof(float3)));
    	CudaSafeCall( cudaMemcpy(deviceClusters, clusters, CLUSTER_COUNT*sizeof(float3), cudaMemcpyHostToDevice));

		printf("Initialized:\n");
    }
    initialized++;


	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);

    do {
        *numberOfPointsThatChangeCluster = 0;

        findNearestClusterAndUpdateMembership
        	<<< grid, block >>>
        	(dptr, deviceClusters, membership, numberOfPointsThatChangeCluster);
        CudaCheckError();
        calculateNewClustersPositions
        	<<< grid, block >>>
        	(dptr, deviceClusters, membership);
        CudaCheckError();

    } while (*numberOfPointsThatChangeCluster/(float)MESH_SIZE > OBJECTS_CLUSTER_CHANGE_THRESHOLD && loop++ < 500);


    *loop_iterations = loop + 1;
}

/*-----------------------------------------------------------------------------------------------------*/

float randFloat(float LO, float HI)
{
	return LO + (float)rand()/((float)RAND_MAX/(HI-LO));
}

void prepare_positions(float time)
{
	for (int index = 0;index<MESH_SIZE;index++)	{

		float u = (index / width) / (float) width;
		float v = (index % width) / (float) height;
	    u = u*2-1;
	    v = v*2-1;

	    // calculate simple sine wave pattern
	    float freq = 4.0f;
	    //float w = sinf(u*freq+time) * cosf(v*freq+time);
	    //float w = sinf(u*freq+time) * sinf(v*freq+time);
	    float w = sinf(freq*sqrtf(u*u + v*v)+time);
	    //float w = (u*u-v*v) * sinf(u+time);
		dptr[index] = make_float3(u, v, w);
	}
}

int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	CudaSafeCall(cudaMallocHost(&dptr, MESH_SIZE*sizeof(float3)));

	printf("starting...\n");

	runProgram(argc, argv, ref_file);

	printf("completed, returned %s\n", (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "CPU: %3.1f FPS \t(X:%d\t Y:%d)", avgFPS, mouse_old_x, mouse_old_y);
	glutSetWindowTitle(fps);
}

bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("K-Means");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
	glewInit();

	if (! glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}

void prepareCuda()
{
	prepare_positions(0);
}


void runKmeans()
{
	//launch_kernel(dptr, width, height, speed);
	if (membership == NULL) {
		CudaSafeCall(cudaMallocHost(&membership, MESH_SIZE*sizeof(int)));
	}

	static double t;
	t += 0.01;
	prepare_positions(t);

	int loops;

	kmeans(membership, &loops);
}

bool runProgram(int argc, char **argv, char *ref_file)
{
	prepareCuda();
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv)) {
		return false;
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);


	// start rendering mainloop
	glutMainLoop();
	atexit(cleanup);

	return true;
}

void setGLColorForCluster(int index)
{
	if (membership[index] == 0)
		glColor3f( 1, 0, 0 );
	if (membership[index] == 1)
		glColor3f( 0, 1, 0 );
	if (membership[index] == 2)
		glColor3f( 0, 0.5, 1 );
	if (membership[index] == 3)
			glColor3f( 0, 1, 1 );
	if (membership[index] == 4)
			glColor3f( 1, 1, 0 );
	if (membership[index] == 5)
			glColor3f( 0.5, 1, 0 );
}

void display()
{
	sdkStartTimer(&timer);

	runKmeans();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);

	glColor3f( 1, 1, 1 );
	glutWireCube (2.0);


	glBegin( GL_POINTS );
	for ( int i = 0; i < MESH_SIZE; ++i )
	{
		setGLColorForCluster(i);
		glPointSize(5);
		glVertex3f( dptr[i].x, dptr[i].y, dptr[i].z );
	}
	glEnd();
	glFinish();
	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
	sdkDeleteTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            exit(EXIT_SUCCESS);
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
