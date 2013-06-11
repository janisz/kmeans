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

////////////////////////////////////////////////////////////////////////////////

#define MESH_SIZE (width*height)

const unsigned int window_width = 800;
const unsigned int window_height = 800;

const unsigned int width = 200;
const unsigned int height = 200;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

//Table containing velocity of all particles. It is pointer to device memory
float2 *speed = NULL;
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
inline __host__ __device__ float distance(float3 pt1, float3 pt2)
{
	float3 v = pt2 - pt1;
	return sqrt(dot(v,v));
}


/*-----------------------------------------------------------------------------------------------------*/

__host__ __device__ inline
float euclidDistance(const float3 coord1, const float3 coord2)
{
    float ans=0.0;

    ans += (coord1.x-coord2.x) * (coord1.x - coord2.x);
    ans += (coord1.y-coord2.y) * (coord1.y - coord2.y);
    ans += (coord1.z-coord2.z) * (coord1.z - coord2.z);

    return ans;
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__golobal__
void findNearestCluster( int numClusters, 	/* no. clusters */
						 int numObjs, 		/* no. objects */
                         float3 *objects, 	/* [numClusters] */
                         float3 *clusters, 	/* [numClusters] */
                         int *membership, 	/* [numObjs] */
					 )
{
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if (objectId < numObjs) {
    	int index = 0;
		float dist, min_dist;

		/* find the cluster id that has min distance to object */
		min_dist = euclidDistance(object, clusters[0]);

		for (int i=1; i<numClusters; i++) {
			dist = euclidDistance(objects[objectId], clusters[i]);
			/* no need square root */
			if (dist < min_dist) { /* find the min and its array index */
				min_dist = dist;
				index = i;
			}
		}
		membership[objectId] = index;
    }
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords] */
void kmeans(float3 *objects, /* in: [numObjs] */
                   int numObjs, /* no. objects */
                   int numClusters, /* no. clusters */
                   float threshold, /* % objects change membership */
                   int *membership, /* out: [numObjs] */
                   int *loop_iterations)
{
    int i, index, loop=0;
    int newClusterSize[numClusters];
    float delta; /* % of objects change their clusters */
    float3 clusters[numClusters];
    float3 newClusters[numClusters];

    //printf("/* pick first numClusters elements of objects[] as initial cluster centers*/\n");
    for (i=0; i<numClusters; i++)
		clusters[i] = objects[i];

    //printf("/* initialize membership[] */\n");
    for (i=0; i<numObjs; i++) membership[i] = -1;

	//printf("/* need to initialize newClusterSize and newClusters[0] to all 0 */\n");
    for (int i=0; i<numClusters; i++) {
    	newClusterSize[i] = 0;
    }

    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {

			//printf("/* find the array index of nearest cluster center */\n");
            index = findNearestCluster(numClusters, objects[i], clusters);

			//printf("/* if membership changes, increase delta by 1 */\n");
            if (membership[i] != index) delta += 1.0;

			//printf("/* assign the membership to object i */\n");
            membership[i] = index;

			//printf("/* update new cluster centers : sum of objects located within */\n");
            newClusterSize[index]++;
			newClusters[index] = newClusters[index] + objects[i];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
			if (newClusterSize[i] > 0)
				clusters[i] = newClusters[i] / newClusterSize[i];
			newClusters[i] = make_float3(0, 0, 0);
            newClusterSize[i] = 0; /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;
}

/*-----------------------------------------------------------------------------------------------------*/

float randFloat(float LO, float HI)
{
	return LO + (float)rand()/((float)RAND_MAX/(HI-LO));
}

void prepare_positions(float3 *pos, float time)
{
	for (int index = 0;index<MESH_SIZE;index++)	{

		float u = (index / width) / (float) width;
		float v = (index % width) / (float) height;
	    u = u*2-1;
	    v = v*2-1;

	    // calculate simple sine wave pattern
	    float freq = 4.0f;
	    float w = sinf(u*freq+time) * sinf(v*freq+time);

		pos[index] = make_float3(u, v, w);
	}
}

int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	dptr = new float3[MESH_SIZE];

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
	prepare_positions(dptr, 0);
}

int *membership = NULL;
#define CLUSTER_COUNT 5
#define OBJECTS_CLUSTER_CHANGE_THRESHOLD 0.1
void runKmeans()
{
	//launch_kernel(dptr, width, height, speed);
	if (membership == NULL) {
		membership = new int[MESH_SIZE];
	}

	static double t;
	t += 0.01;
	prepare_positions(dptr, t);

	int loops;

	kmeans(dptr, /* in: [numObjs][numCoords] */
			   MESH_SIZE, /* no. objects */
			   CLUSTER_COUNT,
			   OBJECTS_CLUSTER_CHANGE_THRESHOLD, /* % objects change membership */
			   membership, /* out: [numObjs] */
			   &loops);
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
