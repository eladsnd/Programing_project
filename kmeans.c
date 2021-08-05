#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int min_ind(double ** centroids, double * vec,int K,int d);
static double * mean(double * vec, int obs,int d);
static double has_changed(double *arr1, double *arr2, int d);

/*
    input:    array - an array from python
              n - int: will be number of observations or number of centroids
              d - dimension of observations
    output:   c_array - C array that contains all items from python array
*/

static double** initialize(PyObject * array , Py_ssize_t n ,Py_ssize_t d)
{
    Py_ssize_t i,j;
    PyObject *item,*list;
    int m;
    double** c_array = (double **)calloc(n, sizeof(double *));
    if (c_array == NULL){
        printf("allocation problem");
        for (m = 0; m < n; m++) {
            free(c_array[m]);
        }
        free(c_array);
        exit(-1);
    }
    for(i=0;i<n;i++) {
        c_array[i]=(double* )calloc(d, sizeof(double ));
        list = PyList_GetItem(array, i);
        if (!PyList_Check(list)) continue;
        for (j = 0; j < d; j++) {
            item = PyList_GetItem(list, j);
            if (!PyFloat_Check(item)) continue;
            c_array[i][j]=PyFloat_AsDouble(item);
        }
    }
    return c_array;
}


/*
    input:    K - number of clusters
              N - number of observations
              d - dimension of observations
              MAX_ITER - constant 300
              centroids_lst - initial list from python
              data - all pf the observations
    output:   dest - C array that tells us what the clusters of observation by the cluster's index
    remarks: this is the "main" function. given all the data it computes us the final clusters according to Kmeans++ algorithm
*/

static int* operations(int K,int N,int d,int MAX_ITER,double **centroids_lst,double **data){
    int a,i,j,m,n,g,vec_index,iteration=0,min_index,changed=1,l,h,y;
    double **sum_vec;
    double sumsum=0;
    int *num_of_obs, *dest;
    if (K > N || MAX_ITER < 1 || d < 1 || K < 1 || N < 1) {
        printf("Invalid Input");
        exit(-1);
    }
    sum_vec = (double **) malloc(K * sizeof(double *));
    if (sum_vec == NULL){
        printf("allocation problem");
        for (a = 0; a < K; a++) {
            free(sum_vec[a]);
        }
        free(sum_vec);
        exit(-1);
    }
    for (g = 0; g < K; g++) {
        sum_vec[g] = (double *) calloc(d, sizeof(double));
        if (sum_vec[g] == NULL) {
            printf("allocation problem");
            for (a = 0; a < K; a++) {
                free(sum_vec[a]);
            }
            free(sum_vec);
            exit(-1);
        }
    }
    num_of_obs = (int *) calloc(K, sizeof(int *));
    if (num_of_obs == NULL){
        printf("allocation problem");
        for (a = 0; a < K; a++) {
            free(sum_vec[a]);
        }
        free(sum_vec);
        free (num_of_obs);
        exit(-1);
    }
    dest = (int *) calloc(N, sizeof(int *));
    if (dest == NULL){
        printf("allocation problem");
        for (a = 0; a < K; a++) {
            free(sum_vec[a]);
        }
        free(sum_vec);
        free (num_of_obs);
        free (dest);
        exit(-1);
    }
    while (iteration < MAX_ITER && changed==1) {
        for (vec_index = 0; vec_index < N; vec_index++) {
            min_index = min_ind(centroids_lst, data[vec_index], K, d);
            num_of_obs[min_index] += 1;
            for (i = 0; i < d; i++) {
                sum_vec[min_index][i] = sum_vec[min_index][i] + data[vec_index][i];
            }
            dest[vec_index]= min_index;
        }
        changed = 1;
        sumsum=0;
        for (j = 0; j < K; j++) {
            double *temp_centroid = mean(sum_vec[j], num_of_obs[j],d);
              sumsum=sumsum+has_changed(temp_centroid,centroids_lst[j],d);

            for (l = 0; l < d; l++) {
                centroids_lst[j][l] = temp_centroid[l];
            }
        }
        if(sumsum<(0.1))
          {

            changed = 0;
          }
        for (h = 0; h < K; h++) {
            num_of_obs[h] = 0;
            for(y=0;y<d;y++){
                sum_vec[h][y] = 0;
            }
        }
        iteration += 1;
    }
    free(num_of_obs);
    for (m = 0; m < N; m++) {
        free(data[m]);
        if (m < K) {
            free(sum_vec[m]);
        }
    }
    free(data);
    free(sum_vec);
    for (n = 0; n < K; n++) {
        free(centroids_lst[n]);
    }
    free(centroids_lst);
    return dest;
}

/*
    input:    vec - C vector size d
              obs - number of observations
              d - dimension of observations
    output:   vec - C array
    remarks:  computes the mean vector from all vectors belongs to some clusters
*/

static double * mean(double * vec, int obs,int d){
    int i;
    for(i = 0; i < d; i++){
        vec[i] = (vec[i] / obs);
    }
    return vec;
}



/*
    input:    arr1- C array size d
              arr2- C array size d
              d - dimension of observations
    output:   sum - C array
    remarks:  computes the vector sum such that sum[i]=(arr1[i] - arr2[i])*(arr1[i] - arr2[i])
*/

 static double has_changed(double *arr1, double *arr2, int d){
    int i;
    double sum;
    sum=0;
   for (i = 0; i < d; i++) {
        sum += (arr1[i] - arr2[i])*(arr1[i] - arr2[i]);
    }
    return sum;
}

/*
    input:    centroids- the centroids in process
              vec- some observation
              K - number of clusters
              d - dimension of observations
    output:   min_index - C int
    remarks:  compute the index of the centroid that vec is closest to
*/

static int min_ind(double **centroids, double *vec,int K,int d) {
    int min_index;
    double min_distance;
    double dist;
    int i;
    min_index = -1;
    min_distance = 0;
    for (i = 0; i < K; i++) {
        dist = has_changed(centroids[i], vec,d);
        if ((dist < min_distance) || (i == 0)) {
            min_index = i;
            min_distance = dist;
        }
    }
    return min_index;
}

/*
    input:    self
              args - k, n, d, MAX_ITER, data, centroids from python
    output:   dest_py - python object that contains the clusters
    remarks:  accept args from python' compute rhe clusters with "operation" function
              and returns pyObject that contains the clusters
*/

static PyObject* kmeans_pp(PyObject *self, PyObject *args)
{
    int k,n,d,MAX_ITER;
    PyObject *data,*centroids,*item, *dest_py;
    int* dest;
    int i,j,place;
    if(!PyArg_ParseTuple(args, "iiiiOO", &k, &n, &d, &MAX_ITER,&data,&centroids)) {
        return NULL;
    }
    if(!PyList_Check(data)){
        return NULL;
    }

    if(!PyList_Check(centroids)){
        return NULL;
    }
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(data, i);
        if (!PyList_Check(item)){  /* We only print lists */
            continue;
        }
    }
    for (i = 0; i < k; i++) {
        item = PyList_GetItem(centroids, i);
        if (!PyList_Check(item)){  /* We only print lists */
            continue;
        }
    }
    double **points = initialize(data,n,d);
    double **my_centroids = initialize(centroids,k,d);
    dest= operations(k,n,d,MAX_ITER,my_centroids,points);
    dest_py = PyList_New(n);
    if (dest_py == NULL){
        printf("pyobject problem");
        exit(-1);
    }
    for (j=0; j<n; j++){
        place = dest[j];
        PyList_SetItem(dest_py, j, Py_BuildValue("i", place));
    }
    free(dest);
    return dest_py;
}



/*
 * way to creates a variable with docstring_kmeans_pp that can be used in docstrings.
 * If Python is built without docstrings, the value will be empty.
 */

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }
PyDoc_STRVAR(docstring_kmeans_pp, "give me some vars i need-  k,n,d,MAX_ITER,data,centroids");

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */

static PyMethodDef _methods[] = {
        {"kmeans_pp", (PyCFunction) kmeans_pp, METH_VARARGS, docstring_kmeans_pp},
        {NULL, NULL, 0, NULL}   /* sentinel */
};

/* This initiates the module using the above definitions. */

static struct PyModuleDef _moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        _methods
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}
