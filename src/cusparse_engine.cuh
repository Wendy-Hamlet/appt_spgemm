#pragma once
// cuSPARSE SpGEMM path exposed as run_cusparse(task, device CSR arrays) -> EngResult.
// Reuses CK/EngResult/checksum from spgemm_engine.cuh (include that first).
#include <cusparse.h>
#define SP(x) do{ cusparseStatus_t s=(x); if(s){fprintf(stderr,"CUSPARSE %s:%d status=%d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

// build A^T (CSR) = CSC of A via cusparseCsr2cscEx2; caller frees.
static void cus_transpose(cusparseHandle_t h,int n,int64_t nnz,const int* Aoff,const int* Acol,
                          const double* Aval,int** Toff,int** Tcol,double** Tval){
    CK(cudaMalloc(Toff,(n+1)*sizeof(int))); CK(cudaMalloc(Tcol,nnz*sizeof(int))); CK(cudaMalloc(Tval,nnz*sizeof(double)));
    size_t bs=0;
    SP(cusparseCsr2cscEx2_bufferSize(h,n,n,nnz,Aval,Aoff,Acol,*Tval,*Toff,*Tcol,CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,&bs));
    void* buf=nullptr; CK(cudaMalloc(&buf,bs));
    SP(cusparseCsr2cscEx2(h,n,n,nnz,Aval,Aoff,Acol,*Tval,*Toff,*Tcol,CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,buf));
    cudaFree(buf);
}
// one full SpGEMM C=A*B; returns nnz; if want copies C values to host for checksum.
static int64_t cus_spgemm(cusparseHandle_t h,int An,int64_t Annz,const int*Aoff,const int*Acol,const double*Aval,
                          int Bn,int64_t Bnnz,const int*Boff,const int*Bcol,const double*Bval,
                          bool want,std::vector<double>* Cout){
    cusparseSpMatDescr_t mA,mB,mC;
    SP(cusparseCreateCsr(&mA,An,An,Annz,(void*)Aoff,(void*)Acol,(void*)Aval,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    SP(cusparseCreateCsr(&mB,Bn,Bn,Bnnz,(void*)Boff,(void*)Bcol,(void*)Bval,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    int* dCoff=nullptr; CK(cudaMalloc(&dCoff,(An+1)*sizeof(int)));
    SP(cusparseCreateCsr(&mC,An,Bn,0,dCoff,nullptr,nullptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    cusparseOperation_t op=CUSPARSE_OPERATION_NON_TRANSPOSE; double al=1.0,be=0.0;
    cusparseSpGEMMDescr_t d; SP(cusparseSpGEMM_createDescr(&d));
    size_t b1=0; void* db1=nullptr;
    SP(cusparseSpGEMM_workEstimation(h,op,op,&al,mA,mB,&be,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,d,&b1,nullptr));
    CK(cudaMalloc(&db1,b1)); SP(cusparseSpGEMM_workEstimation(h,op,op,&al,mA,mB,&be,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,d,&b1,db1));
    size_t b2=0; void* db2=nullptr;
    SP(cusparseSpGEMM_compute(h,op,op,&al,mA,mB,&be,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,d,&b2,nullptr));
    CK(cudaMalloc(&db2,b2)); SP(cusparseSpGEMM_compute(h,op,op,&al,mA,mB,&be,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,d,&b2,db2));
    int64_t Cr,Cc,Cnnz; SP(cusparseSpMatGetSize(mC,&Cr,&Cc,&Cnnz));
    int* dCcol=nullptr; double* dCval=nullptr; CK(cudaMalloc(&dCcol,Cnnz*sizeof(int))); CK(cudaMalloc(&dCval,Cnnz*sizeof(double)));
    SP(cusparseCsrSetPointers(mC,dCoff,dCcol,dCval));
    SP(cusparseSpGEMM_copy(h,op,op,&al,mA,mB,&be,mC,CUDA_R_64F,CUSPARSE_SPGEMM_DEFAULT,d));
    if(want&&Cout){ Cout->resize(Cnnz); CK(cudaMemcpy(Cout->data(),dCval,Cnnz*sizeof(double),cudaMemcpyDeviceToHost)); }
    cusparseSpGEMM_destroyDescr(d); cusparseDestroySpMat(mA);cusparseDestroySpMat(mB);cusparseDestroySpMat(mC);
    cudaFree(db1);cudaFree(db2);cudaFree(dCoff);cudaFree(dCcol);cudaFree(dCval);
    return Cnnz;
}
static EngResult run_cusparse(const char* task,const int* dOff,const int* dCol,const double* dVal,
                              int n,int64_t nnz,int reps){
    bool AAt=(strcmp(task,"AAt")==0)||(strcmp(task,"AAtS")==0);
    cusparseHandle_t h; SP(cusparseCreate(&h));
    int *Boff=(int*)dOff,*Bcol=(int*)dCol; double *Bval=(double*)dVal; int64_t Bnnz=nnz;
    // warmup + correctness (build transpose inside timed region for fairness)
    auto once=[&](bool want,std::vector<double>* C)->int64_t{
        int *To=nullptr,*Tc=nullptr; double* Tv=nullptr; int64_t r;
        if(AAt){ cus_transpose(h,n,nnz,dOff,dCol,dVal,&To,&Tc,&Tv);
            r=cus_spgemm(h,n,nnz,dOff,dCol,dVal,n,nnz,To,Tc,Tv,want,C); cudaFree(To);cudaFree(Tc);cudaFree(Tv); }
        else r=cus_spgemm(h,n,nnz,dOff,dCol,dVal,n,nnz,Boff,Bcol,Bval,want,C);
        return r;
    };
    std::vector<double> C; int64_t Cnnz=once(true,&C);
    double s,as,sq; checksum(C,s,as,sq);
    for(int i=0;i<2;i++) once(false,nullptr);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); cudaEventRecord(e0);
    for(int r=0;r<reps;r++) once(false,nullptr);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=reps;
    cusparseDestroy(h);
    return EngResult{ms,Cnnz,s,as,sq};
}
