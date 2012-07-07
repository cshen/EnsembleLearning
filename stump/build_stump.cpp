/*
 * File Type:     C/C++
 * Author:        Peng Wang {wangpengnorman@gmail.com}, Chunhua Shen
 * Creation:      Thursday 06/03/2010 11:11.
 * Last Revision: Tuesday 31/05/2011 14:41.
 * OpenMP enabled, May 2011, CS
 */


#define NUM_THREADS_OMP  4



#include "mex.h"
#include <omp.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
	int             llabel;
	int             rlabel;
	double          lerror;
	double          rerror;
	double          min_err;
	double          threshold;
	int             dim_idx;
}               StumpClassifier;



bool
TrainSingleDim(const double *x,
	       const double *w,
	       const int *idx,
	       int num, double sum_w,
	       StumpClassifier * stump,
	       int dim_idx)
{
	bool            found = false;
	double          threshold = 0;
	int             llabel = 0, rlabel = 0;
	double          min_err = DBL_MAX;
	double          cur_err = sum_w;

	/* double pre_x = x[(int)(idx[0]-1)]; */

	double          pre_x = x[0];
	double          pre_w = w[idx[0] - 1];
    
//     double zero_w_sum(0);
//     double non_zero_w_sum(0);
//     if (pre_x==0){
//         zero_w_sum+=pre_x;
//     }else{
//         non_zero_w_sum+=pre_w;
//     }
    
	int             i;
	for (i = 1; i < num; i++) {
		int             cur_idx = idx[i] - 1;
		/* double cur_x = x[cur_idx]; */
		double          cur_x = x[i];

               
		cur_err -= 2 * pre_w;

		if (cur_x != pre_x) {
			if (cur_err < min_err) {
				found = true;
				threshold = 0.5 * (cur_x + pre_x);
				llabel = -1;
				rlabel = +1;
				min_err = cur_err;
			}
			if (-1.f * cur_err < min_err) {
				found = true;
				threshold = 0.5 * (cur_x + pre_x);
				llabel = +1;
				rlabel = -1;
				min_err = -1.f * cur_err;
			}
		}
		pre_x = cur_x;
		pre_w = w[cur_idx];
        
        
//         if (cur_x==0){
//             zero_w_sum+=pre_w;  
//         }else{
//             non_zero_w_sum+=pre_w;
//         }
//         printf("onedim: %d %f %f \n",cur_idx, cur_err, min_err);
        
	}
    
//     printf("%f %f %f \n",sum_w, non_zero_w_sum, zero_w_sum);

//     mexEvalString("pause();"); 

	if (found) {
		stump->llabel = llabel;
		stump->rlabel = rlabel;
		stump->lerror = 0;
		stump->rerror = 0;
		stump->min_err = min_err;
		stump->threshold = threshold;
	}
	return found;

}




void
TrainMultiDim(const double *sorted_data,
	      const double *weights,
	      const int *sorted_indices,
	      double sum_w,
	      int num, int dim,
	      StumpClassifier * opt_stump)
{
	double          min_err = DBL_MAX;
	const double   *w = weights;


	StumpClassifier *stump;
	stump = (StumpClassifier *) malloc(sizeof(StumpClassifier) * dim);

    bool * found;
    found = (bool *) malloc (sizeof(bool) * dim);

    int             i;


//     (void)omp_set_num_threads( NUM_THREADS_OMP );


//     #pragma omp parallel for
    for (i = 0; i < dim; i++) {
    
//        mexPrintf("Num threads %d, thread ID %d.\n", omp_get_num_threads(), omp_get_thread_num());
		
        const double   *x   = &(sorted_data[i * num]);
		const int      *idx = &(sorted_indices[i * num]);

		found[i] = TrainSingleDim(x, w, idx, num, sum_w, &stump[i], i);

	}



    for (i = 0; i < dim; i++) {
        if (found[i] && (min_err > stump[i].min_err))
        {
            opt_stump->llabel    = stump[i].llabel;
            opt_stump->rlabel    = stump[i].rlabel;
            opt_stump->lerror    = stump[i].lerror;
            opt_stump->rerror    = stump[i].rerror;
            opt_stump->min_err   = stump[i].min_err;
            opt_stump->threshold = stump[i].threshold;
            opt_stump->dim_idx   = i;
            min_err = stump[i].min_err;
        }
        
//         printf("%d %f %f \n",i, stump[i].min_err, stump[i].threshold);
    }


    free(stump);
    free(found);

}




/*
 * function [w, b, dim_idx, err] = stump_train_multi_dim_fast(sorted_data,
 * weights, sorted_indices, sum_w) size(sorted_data) = [dim, num];
 * size(weights) = [1, num]; size(sorted_indices) = [dim, num];
 */
void
mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{


	/* inputs */
	/* printf("inputs\n"); */
	/* mxAssert(nrhs == 4, "nrhs != 4"); */

	const double   *sorted_data = mxGetPr(prhs[0]);
	int             num = mxGetM(prhs[0]);
	int             dim = mxGetN(prhs[0]);
	const double   *weights = mxGetPr(prhs[1]);
	const int      *sorted_indices = (const int *) mxGetData(prhs[2]);
	double          sum_w = mxGetScalar(prhs[3]);


	if (nrhs != 4) {
		mexErrMsgTxt("Four inputs required.");
	}

	        //printf("num: %d dim: %d, sum_w: %g, nrhs %d, temp %d\n", num, dim, sum_w,nrhs,mxGetM(prhs[1]));

		mxAssert(num == mxGetM(prhs[1]), "num != mxGetM(prhs[1])");
		mxAssert(num == mxGetM(prhs[2]), "num != mxGetM(prhs[2])");
		mxAssert(dim == mxGetN(prhs[2]), "num != mxGetN(prhs[2])");

		/*
	        printf("num: %d dim: %d, sum_w: %g\n", num, dim, sum_w);
	        printf("weights: \n");
	        int k;
	        for(k = 0; k < 100; k++)
	        {
	          printf("%g ", weights[k]);
	        }
	        printf("\n");
	        */

		/* outputs */
		/* printf("outputs\n"); */
		/* mxAssert(nlhs == 4, "nlhs != 4"); */
		if (nlhs != 4)
			mexErrMsgTxt("Four outputs required.");


			plhs[0] = mxCreateDoubleScalar(0);
			plhs[1] = mxCreateDoubleScalar(0);
			plhs[2] = mxCreateDoubleScalar(0);
			plhs[3] = mxCreateDoubleScalar(0);

			double         *w = mxGetPr(plhs[0]);
			double         *b = mxGetPr(plhs[1]);
			double         *dim_idx = mxGetPr(plhs[2]);
			double         *err = mxGetPr(plhs[3]);
            
			/* opetations */
			StumpClassifier opt_stump;
                        
            //init
            opt_stump.llabel=0;
            opt_stump.rlabel=0;
            opt_stump.dim_idx=-1;
            opt_stump.threshold=0;
            opt_stump.min_err=DBL_MAX;
            
            
			TrainMultiDim(sorted_data, weights, sorted_indices, sum_w, num, dim, &opt_stump);

			/* assignments */
			/* prd_val = sign(-llabel * (x - threshold)) */
			/* printf("assignments"); */
			*w = (double) -1.0 * opt_stump.llabel;
			*b = (double) opt_stump.llabel * opt_stump.threshold;
			*dim_idx = (double) opt_stump.dim_idx + 1;
			*err = (double) opt_stump.min_err;
		
		/* EOF */
}



