<div style="text-align: center;"><img src="https://mma.prnewswire.com/media/1167265/Tel_Aviv_University_Logo.jpg?p=facebook" width="50%" height="50%" alt="TAU"></div>

# TAU Software Project Course
## Final Project
### Submitted by
* **Elad Ben Shabat** &ensp;
* **Shir Basa**&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; 

### Project Modules
#### K-means C-API
The following files are required for running the K-means++ algorithm (using Python's C-API)
* `kmeans.c`
* `kmeans.h`
* `kmeans_pp.py`
* `setup.py`


#### Spectral Clustering Algorithm
* `main.py` — project's entering point
* `ops.py` — opertaions on matrices such as creating the normalized laplacian matrix, qr decomposition, MGS and more  
* `kmeans_pp.py` — K-means++ algorithm implementation
* `tasks.py` — invoke tasks
* `utils.py` — exporting output result files
* `visual.py` — exporting visual pdf file for comapring K-means and Spectral Clustering algorithms results


### How to run?
Use the following command:
```bash
python3.8.5 -m invoke run -k [k] -n [n] [--[no-]Random]
```

Output files:
* `data.txt` — randomly generated data;
* `clusters.txt` — computed clusters for both algorithms
* `clusters.pdf` — graphical visualization comparing K-means and Spectral Clustering algorithms



