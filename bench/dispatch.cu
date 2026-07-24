// Online structure-aware SpGEMM dispatcher.
#include <cmath>
// At runtime: compute cheap structural descriptors (deg_cv, fill) from the CSR,
// apply a fixed decision rule (learned offline, dominated by row-imbalance
// deg_cv), then run the chosen implementation {ours, cuSPARSE}. Reports the
// choice, the descriptor-computation overhead, and the SpGEMM time -- so the
// online cost (overhead + run) is fully accounted.
#include "../src/spgemm_engine.cuh"
#include "../src/cusparse_engine.cuh"
#include <chrono>

// Decision rule (offline depth-3 tree, min_samples_leaf=8; feature = deg_cv, fill, nnz):
//   use ours iff (dcv<=0.63 && fill<=53.23) || (dcv>0.63 && nnz>20164 && fill<=47.17)
static bool pick_ours(double dcv,double fill,int64_t nnz){
    if(dcv<=0.63) return fill<=53.23;
    return (nnz>20164 && fill<=47.17);
}

int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s <tag> <bin.csr> <AA|AAt> [reps]\n",argv[0]);return 1;}
    const char* tag=argv[1]; const char* task=argv[3]; int reps=argc>4?atoi(argv[4]):20;
    bool AAt=(strcmp(task,"AAt")==0);
    Csr h=load_bin(argv[2]); int n=h.n; int64_t nnz=h.nnz;

    // ---- structural descriptors (host, timed) ----
    auto t0=std::chrono::high_resolution_clock::now();
    double dmean=(double)nnz/n, var=0;
    for(int i=0;i<n;i++){ double d=h.indptr[i+1]-h.indptr[i]-dmean; var+=d*d; }
    double dcv=std::sqrt(var/n)/(dmean>1e-9?dmean:1e-9);
    double fill;
    if(!AAt){                                   // fill_aa = mean deg of referenced cols
        int64_t acc=0; for(int64_t p=0;p<nnz;p++){int k=h.indices[p]; acc+=h.indptr[k+1]-h.indptr[k];}
        fill=(double)acc/nnz;
    } else {                                    // fill_aat = sum colcount^2 / nnz
        std::vector<int64_t> cc(n,0); for(int64_t p=0;p<nnz;p++) cc[h.indices[p]]++;
        int64_t acc=0; for(int i=0;i<n;i++) acc+=cc[i]*cc[i]; fill=(double)acc/nnz;
    }
    bool ours=pick_ours(dcv,fill,nnz);
    auto t1=std::chrono::high_resolution_clock::now();
    double desc_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    // ---- copy to device, run chosen ----
    int *dOff,*dCol; double *dVal;
    CK(cudaMalloc(&dOff,(n+1)*sizeof(int))); CK(cudaMalloc(&dCol,nnz*sizeof(int))); CK(cudaMalloc(&dVal,nnz*sizeof(double)));
    CK(cudaMemcpy(dOff,h.indptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCol,h.indices.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVal,h.data.data(),nnz*sizeof(double),cudaMemcpyHostToDevice));
    EngResult R = ours ? run_ours(task,dOff,dCol,dVal,n,nnz,reps)
                       : run_cusparse(task,dOff,dCol,dVal,n,nnz,reps);
    printf("DISPATCH,%s,%s,%s,dcv=%.3f,fill=%.1f,C_nnz=%lld,desc_ms=%.4f,run_ms=%.4f,total_ms=%.4f\n",
           tag,task,ours?"ours":"cusparse",dcv,fill,(long long)R.nnz,desc_ms,R.ms,desc_ms+R.ms);
    return 0;
}
