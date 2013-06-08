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

const unsigned int width = 150;
const unsigned int height = 150;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

//Table containing velocity of all particles. It is pointer to device memory
float2 *speed = NULL;
float4 *dptr = NULL;

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

__inline static
float euclid_dist_2(int numdims, /* no. dimensions */
                    float *coord1, /* [numdims] */
                    float *coord2) /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int numClusters, /* no. clusters */
                         int numCoords, /* no. coordinates */
                         float *object, /* [numCoords] */
                         float **clusters) /* [numClusters][numCoords] */
{
    int index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index = i;
        }
    }
    return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords] */
float** seq_kmeans(float **objects, /* in: [numObjs][numCoords] */
                   int numCoords, /* no. features */
                   int numObjs, /* no. objects */
                   int numClusters, /* no. clusters */
                   float threshold, /* % objects change membership */
                   int *membership, /* out: [numObjs] */
                   int *loop_iterations)
{
    int i, j, index, loop=0;
    int *newClusterSize; /* [numClusters]: no. objects assigned in each
new cluster */
    float delta; /* % of objects change their clusters */
    float **clusters; /* out: [numClusters][numCoords] */
    float **newClusters; /* [numClusters][numCoords] */

    //printf("/* allocate a 2D space for returning variable clusters[] (coordinates of cluster centers) */\n");
    clusters = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*) malloc(numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    //printf("/* pick first numClusters elements of objects[] as initial cluster centers*/\n");
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];

    //printf("/* initialize membership[] */\n");
    for (i=0; i<numObjs; i++) membership[i] = -1;

	//printf("/* need to initialize newClusterSize and newClusters[0] to all 0 */\n");
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*) calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {

			//printf("/* find the array index of nearest cluster center */\n");
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);

			//printf("/* if membership changes, increase delta by 1 */\n");
            if (membership[i] != index) delta += 1.0;

			//printf("/* assign the membership to object i */\n");
            membership[i] = index;

			//printf("/* update new cluster centers : sum of objects located within */\n");
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0; /* set back to 0 */
            }
            newClusterSize[i] = 0; /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

/*-----------------------------------------------------------------------------------------------------*/

float randFloat(float LO, float HI)
{
	return LO + (float)rand()/((float)RAND_MAX/(HI-LO));
}

void prepare_kernel(float4 *pos, float time)
{
	for (int index = 0;index<MESH_SIZE;index++)	{
		// calculate uv coordinates
		float u = (index / width) / (float) width;
		float v = (index % width) / (float) height;
	    u = u*2-1;
	    v = v*2-1;
	    // calculate simple sine wave pattern
	    float freq = 4.0f;

	    float w = sinf(u*freq+time) * sinf(v*freq+time);

		// write output vertex
		pos[index] = make_float4(u, v, w, 0);

	}
}

int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

	dptr = new float4[MESH_SIZE];

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
	glutCreateWindow("Cuda GL Interop (VBO)");
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
	prepare_kernel(dptr, 0);
}

int *membership = NULL;
float** obj = NULL;

void runKmeans()
{
	//launch_kernel(dptr, width, height, speed);
	if (membership == NULL) {
		membership = new int[height*width];
	}

	static double t;
	t += 0.01;
	prepare_kernel(dptr, t);

	if (obj == NULL) {
		obj = new float*[height*width];
	}

	for (int i=0;i<height*width;i++) {
		obj[i] = (float*)(&dptr[i]);
	}
	int loops;

	seq_kmeans(obj, /* in: [numObjs][numCoords] */
			   4, 	/* no. features */
			   width * height, /* no. objects */
			   3, 	/* no. clusters */
			   0.01, /* % objects change membership */
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
