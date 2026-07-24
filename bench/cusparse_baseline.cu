// cuSPARSE SpGEMM baseline for the APPT 2026 challenge.
//   task A*A   : C = A * A            (both NON_TRANSPOSE)
//   task A*A^T : C = A * A^T          (A^T built explicitly via csr2csc)
// cuSPARSE generic SpGEMM does not support transpose ops, so A^T is
// materialized as CSR (= CSC of A) first, which is the standard approach.
//
// Compiled for sm_80 (A100) even on H100 so numbers port to A100.
// Output: CSV lines  tag,task,n,nnz,C_nnz,ms,gflops_ignored,sum,abssum,sq
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define SP(x) do{ cusparseStatus_t s=(x); if(s){fprintf(stderr,"CUSPARSE %s:%d status=%d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

struct Csr { int n; int64_t nnz; std::vector<int> indptr, indices; std::vector<double> data; };

static Csr load_bin(const char* path){
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(1);}
    char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"CSR1",4)){fprintf(stderr,"bad magic %s\n",path);exit(1);}
    Csr c; int pad; if(fread(&c.n,4,1,f)!=1) exit(1); if(fread(&pad,4,1,f)!=1) exit(1);
    if(fread(&c.nnz,8,1,f)!=1) exit(1);
    c.indptr.resize(c.n+1); c.indices.resize(c.nnz); c.data.resize(c.nnz);
    if((int64_t)fread(c.indptr.data(),4,c.n+1,f)!=c.n+1) exit(1);
    if((int64_t)fread(c.indices.data(),4,c.nnz,f)!=c.nnz) exit(1);
    if((int64_t)fread(c.data.data(),8,c.nnz,f)!=c.nnz) exit(1);
    fclose(f); return c;
}

// device CSR
struct DCsr { int n,m; int64_t nnz; int *off=nullptr,*col=nullptr; double *val=nullptr; };

static DCsr to_dev(const Csr& h){
    DCsr d; d.n=h.n; d.m=h.n; d.nnz=h.nnz;
    CK(cudaMalloc(&d.off,(h.n+1)*sizeof(int)));
    CK(cudaMalloc(&d.col,h.nnz*sizeof(int)));
    CK(cudaMalloc(&d.val,h.nnz*sizeof(double)));
    CK(cudaMemcpy(d.off,h.indptr.data(),(h.n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.col,h.indices.data(),h.nnz*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.val,h.data.data(),h.nnz*sizeof(double),cudaMemcpyHostToDevice));
    return d;
}
static void free_dev(DCsr&d){ if(d.off)cudaFree(d.off); if(d.col)cudaFree(d.col); if(d.val)cudaFree(d.val); d.off=d.col=nullptr; d.val=nullptr; }

// Build A^T (CSR) = CSC of A, via cusparseCsr2cscEx2.
static DCsr transpose(cusparseHandle_t h, const DCsr& A){
    DCsr T; T.n=A.m; T.m=A.n; T.nnz=A.nnz;
    CK(cudaMalloc(&T.off,(T.n+1)*sizeof(int)));
    CK(cudaMalloc(&T.col,A.nnz*sizeof(int)));
    CK(cudaMalloc(&T.val,A.nnz*sizeof(double)));
    size_t bsz=0;
    SP(cusparseCsr2cscEx2_bufferSize(h,A.n,A.m,A.nnz,A.val,A.off,A.col,
        T.val,T.off,T.col,CUDA_R_64F,CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,&bsz));
    void* buf=nullptr; CK(cudaMalloc(&buf,bsz));
    SP(cusparseCsr2cscEx2(h,A.n,A.m,A.nnz,A.val,A.off,A.col,
        T.val,T.off,T.col,CUDA_R_64F,CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,buf));
    cudaFree(buf); return T;
}

// One SpGEMM C = A*B. Returns C_nnz and (optionally) copies C to host for checksum.
// Times the full symbolic+numeric path.
static double spgemm_once(cusparseHandle_t h, const DCsr& A, const DCsr& B,
                          int64_t* out_nnz, bool want_C, Csr* Chost){
    cusparseSpMatDescr_t mA,mB,mC;
    SP(cusparseCreateCsr(&mA,A.n,A.m,A.nnz,(void*)A.off,(void*)A.col,(void*)A.val,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    SP(cusparseCreateCsr(&mB,B.n,B.m,B.nnz,(void*)B.off,(void*)B.col,(void*)B.val,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    int Cn=A.n, Cm=B.m;
    int* dCoff=nullptr; CK(cudaMalloc(&dCoff,(Cn+1)*sizeof(int)));
    SP(cusparseCreateCsr(&mC,Cn,Cm,0,dCoff,nullptr,nullptr,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    cusparseOperation_t op=CUSPARSE_OPERATION_NON_TRANSPOSE;
    double alpha=1.0, beta=0.0;
    cusparseSpGEMMDescr_t desc; SP(cusparseSpGEMM_createDescr(&desc));
    size_t bs1=0; void* db1=nullptr;
    SP(cusparseSpGEMM_workEstimation(h,op,op,&alpha,mA,mB,&beta,mC,CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,nullptr));
    CK(cudaMalloc(&db1,bs1));
    SP(cusparseSpGEMM_workEstimation(h,op,op,&alpha,mA,mB,&beta,mC,CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,db1));
    size_t bs2=0; void* db2=nullptr;
    SP(cusparseSpGEMM_compute(h,op,op,&alpha,mA,mB,&beta,mC,CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,nullptr));
    CK(cudaMalloc(&db2,bs2));
    SP(cusparseSpGEMM_compute(h,op,op,&alpha,mA,mB,&beta,mC,CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,db2));
    int64_t Crows,Ccols,Cnnz; SP(cusparseSpMatGetSize(mC,&Crows,&Ccols,&Cnnz));
    int* dCcol=nullptr; double* dCval=nullptr;
    CK(cudaMalloc(&dCcol,Cnnz*sizeof(int))); CK(cudaMalloc(&dCval,Cnnz*sizeof(double)));
    SP(cusparseCsrSetPointers(mC,dCoff,dCcol,dCval));
    SP(cusparseSpGEMM_copy(h,op,op,&alpha,mA,mB,&beta,mC,CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,desc));
    *out_nnz=Cnnz;
    if(want_C && Chost){
        Chost->n=Cn; Chost->nnz=Cnnz; Chost->indptr.resize(Cn+1);
        Chost->indices.resize(Cnnz); Chost->data.resize(Cnnz);
        CK(cudaMemcpy(Chost->indptr.data(),dCoff,(Cn+1)*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Chost->indices.data(),dCcol,Cnnz*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Chost->data.data(),dCval,Cnnz*sizeof(double),cudaMemcpyDeviceToHost));
    }
    cusparseSpGEMM_destroyDescr(desc);
    cusparseDestroySpMat(mA); cusparseDestroySpMat(mB); cusparseDestroySpMat(mC);
    cudaFree(db1); cudaFree(db2); cudaFree(dCoff); cudaFree(dCcol); cudaFree(dCval);
    return 0.0;
}

// checksum on host C (order-independent)
static void checksum(const Csr& C, double& s, double& as, double& sq){
    s=as=sq=0; for(int64_t i=0;i<C.nnz;i++){double v=C.data[i]; s+=v; as+=v<0?-v:v; sq+=v*v;}
}

static double bench(cusparseHandle_t h,const DCsr&A,const DCsr&B,int reps,int64_t*nnz){
    // warmup + correctness copy
    Csr C; spgemm_once(h,A,B,nnz,true,&C);
    double s,as,sq; checksum(C,s,as,sq);
    for(int i=0;i<2;i++){ int64_t t; spgemm_once(h,A,B,&t,false,nullptr); }
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int i=0;i<reps;i++){ int64_t t; spgemm_once(h,A,B,&t,false,nullptr); }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    // stash checksum via globals-through-pointer trick: print inline instead
    printf("  __chk s=%.6e as=%.6e sq=%.6e\n",s,as,sq);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms/reps;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <tag> <bin.csr> [reps]\n",argv[0]);return 1;}
    const char* tag=argv[1]; const char* path=argv[2];
    int reps=argc>3?atoi(argv[3]):10;
    Csr h=load_bin(path);
    cusparseHandle_t handle; SP(cusparseCreate(&handle));
    DCsr A=to_dev(h);
    int64_t nnzAA=0, nnzAAt=0;
    double t_aa = bench(handle,A,A,reps,&nnzAA);
    DCsr At=transpose(handle,A);
    double t_aat = bench(handle,A,At,reps,&nnzAAt);
    printf("RESULT,%s,AA,%d,%lld,%lld,%.4f\n",tag,h.n,(long long)h.nnz,(long long)nnzAA,t_aa);
    printf("RESULT,%s,AAt,%d,%lld,%lld,%.4f\n",tag,h.n,(long long)h.nnz,(long long)nnzAAt,t_aat);
    free_dev(A); free_dev(At); cusparseDestroy(handle);
    return 0;
}
