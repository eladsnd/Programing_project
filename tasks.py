from invoke import task, call

"""
task - build

this is set to build a module of kmeans++ 
the module will reactive observations data in a form of NxD array and starting points representing random points
to be used as stating clusters
and will return a vector in witch will be assignments of each observations to a cluster

the task will run build setup 
"""


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


"""
task - run

pre - will run task - build
this is set to run the main function of our code 
the main func will run nsc and kmeans and will show them side by side with their jaccard measure

args: k - the number of clusters , defaulted to 0 
      n - the number of observations , defaulted to 0 
      Random - a boolean value will dictate if the data will be in the sises specified (n,k)
               or if we will use random values dictated by the max capacity of main and will also
               choose clusters number for our algorithms (k-means and nsc)
"""


@task(build)
def run(c, k=0, n=0, Random = True ):
    if Random:
        c.run("python3.8.5 main.py {} {} {}".format(k, n, "--Random"))
    else:
        c.run("python3.8.5 main.py {} {} {}".format(k, n, "--no-Random"))
    print("Done")



