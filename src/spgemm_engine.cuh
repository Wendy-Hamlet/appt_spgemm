#pragma once
// Unified custom hash-based SpGEMM: C = A*B, fp64 (v3 engine generalized).
//   task AA  : B = A            -> C = A*A
//   task AAt : B = A^T (built)  -> C = A*A^T   (row i of C = <row_i(A), row_j(A)>)
// A*A^T reuses the exact same engine with B=A^T, so both tasks share one kernel.
// Per-row-sized shared hash (+ global-arena overflow) + CUB reused-temp sort.
// Full pipeline (incl. transpose for AAt) is timed, fair vs cuSPARSE. sm_80.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static const int SH_CAP=1024, WPB=4;   // rows whose hash fits SH_CAP use shared path

struct Csr { int n; int64_t nnz; std::vector<int> indptr, indices; std::vector<double> data; };
static Csr load_bin(const char* path){
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(1);}
    char m[4]; if(fread(m,1,4,f)!=4||memcmp(m,"CSR1",4)){fprintf(stderr,"bad magic\n");exit(1);}
    Csr c; int pad; if(fread(&c.n,4,1,f)!=1)exit(1); if(fread(&pad,4,1,f)!=1)exit(1);
    if(fread(&c.nnz,8,1,f)!=1)exit(1);
    c.indptr.resize(c.n+1); c.indices.resize(c.nnz); c.data.resize(c.nnz);
    if((int64_t)fread(c.indptr.data(),4,c.n+1,f)!=c.n+1)exit(1);
    if((int64_t)fread(c.indices.data(),4,c.nnz,f)!=c.nnz)exit(1);
    if((int64_t)fread(c.data.data(),8,c.nnz,f)!=c.nnz)exit(1);
    fclose(f); return c;
}
__host__ __device__ static inline int64_t np2l(int64_t x){ if(x<2)return 2; int64_t p=2; while(p<x)p<<=1; return p; }

// ---- transpose A (CSR n x n) -> B = A^T (CSR) ----
__global__ void k_memset_i(int* a,int n,int v){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)a[i]=v;}
__global__ void k_colcount(int64_t nnz,const int* Acol,int* cnt){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p<nnz) atomicAdd(&cnt[Acol[p]],1);
}
__global__ void k_scatter(int n,const int* Aoff,const int* Acol,const double* Aval,
                          int* cursor,int* Bcol,double* Bval){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    for(int p=Aoff[i];p<Aoff[i+1];p++){int k=Acol[p]; int pos=atomicAdd(&cursor[k],1); Bcol[pos]=i; Bval[pos]=Aval[p];}
}

// ---- generalized SpGEMM engine C = A * B ----
__global__ void k_flops(int n,const int* Aoff,const int* Acol,const int* Boff,int64_t* flops){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int s=Aoff[i],e=Aoff[i+1]; int64_t f=0; for(int p=s;p<e;p++){int k=Acol[p]; f+=Boff[k+1]-Boff[k];} flops[i]=f;
}
__global__ void k_caps(int n,const int64_t* flops,int* rcap,int64_t* gcap){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; int64_t f=flops[i];
    if(f==0){rcap[i]=0;gcap[i]=0;return;}
    int64_t want=np2l((f*4)/3+1);            // ~1.33x load factor, power of 2
    if(want<=SH_CAP){rcap[i]=(int)want;gcap[i]=0;}   // fits shared slice -> shared path
    else {rcap[i]=0;gcap[i]=want;}                    // else global-arena path
}
// Pure-atomicCAS open-addressing insert: robust for BOTH shared and global
// memory. A plain pre-read of keys[h] is unsafe in shared memory (the compiler
// may cache it / not observe other lanes' atomicCAS writes), which duplicated
// keys across slots -> over-counted nnz + nondeterminism. Every probe is an
// atomicCAS instead.
__device__ static inline int ins(int* keys,int cap,int key){
    unsigned h=((unsigned)key*2654435761u)&(cap-1);
    while(true){
        int cur=atomicCAS(&keys[h],-1,key);   // claim if empty; else read occupant
        if(cur==-1||cur==key) return (int)h;   // inserted, or already present
        h=(h+1)&(cap-1);
    }
}
__global__ void k_symbolic(int n,const int* Aoff,const int* Acol,const int* Boff,const int* Bcol,
                           const int64_t* flops,const int* rcap,const int64_t* hoff,const int64_t* gcap,
                           int* gkeys,int* rowcnt,int sym){
    extern __shared__ int sh[];
    int wib=threadIdx.x>>5, lane=threadIdx.x&31; int i=(blockIdx.x*WPB)+wib; if(i>=n)return;
    int64_t f=flops[i]; if(f==0){ if(lane==0)rowcnt[i]=0; return; }
    int s=Aoff[i],e=Aoff[i+1], rc=rcap[i];
    int* keys; int cap;
    if(rc>0){ keys=sh+wib*SH_CAP; cap=rc; } else { cap=(int)gcap[i]; keys=gkeys+hoff[i]; }
    for(int t=lane;t<cap;t+=32)keys[t]=-1; __syncwarp();
    for(int p=s;p<e;p++){int k=Acol[p]; int ks=Boff[k],ke=Boff[k+1];
        for(int q=ks+lane;q<ke;q+=32){int j=Bcol[q]; if(sym&&j<i)continue; ins(keys,cap,j);} }
    __syncwarp();
    int cnt=0; for(int t=lane;t<cap;t+=32) if(keys[t]!=-1)cnt++;
    for(int o=16;o>0;o>>=1)cnt+=__shfl_down_sync(0xffffffff,cnt,o);
    if(lane==0)rowcnt[i]=cnt;
}
__global__ void k_numeric(int n,const int* Aoff,const int* Acol,const double* Aval,
                          const int* Boff,const int* Bcol,const double* Bval,
                          const int64_t* flops,const int* rcap,const int64_t* hoff,const int64_t* gcap,
                          int* gkeys,double* gvals,const int* Coff,int64_t* Ckey,double* Cval,int sym){
    extern __shared__ char shm[];
    int wib=threadIdx.x>>5, lane=threadIdx.x&31;
    int* skeys=(int*)shm + wib*SH_CAP;
    double* svals=(double*)((int*)shm + WPB*SH_CAP) + wib*SH_CAP;
    int i=(blockIdx.x*WPB)+wib; if(i>=n)return;
    int64_t f=flops[i]; if(f==0)return;
    int s=Aoff[i],e=Aoff[i+1], wpos=Coff[i], rc=rcap[i];
    int* keys; double* vals; int cap;
    if(rc>0){ keys=skeys; vals=svals; cap=rc; }
    else { cap=(int)gcap[i]; keys=gkeys+hoff[i]; vals=gvals+hoff[i]; }
    for(int t=lane;t<cap;t+=32){keys[t]=-1; vals[t]=0.0;} __syncwarp();
    for(int p=s;p<e;p++){int k=Acol[p]; double aik=Aval[p]; int ks=Boff[k],ke=Boff[k+1];
        for(int q=ks+lane;q<ke;q+=32){int j=Bcol[q]; if(sym&&j<i)continue; int slot=ins(keys,cap,j); atomicAdd(&vals[slot],aik*Bval[q]); } }
    __syncwarp();
    // warp-uniform extract: ALL 32 lanes must reach __ballot_sync every iter,
    // so iterate in blocks of 32 (cap may be <32 or not a multiple of 32).
    for(int base=0;base<cap;base+=32){
        int t=base+lane; int key=(t<cap)?keys[t]:-1;
        unsigned mk=__ballot_sync(0xffffffff,key!=-1);
        int rank=__popc(mk&((1u<<lane)-1));
        if(key!=-1){int pos=wpos+rank; Ckey[pos]=(int64_t)i*n+key; Cval[pos]=vals[t];}
        wpos+=__popc(mk);
    }
}
static void checksum(const std::vector<double>& v,double&s,double&as,double&sq){
    s=as=sq=0; for(double x:v){s+=x;as+=x<0?-x:x;sq+=x*x;}
}
// mirror upper-triangular C (task AAtS) to the full symmetric matrix:
// each off-diagonal (i,j) also emits (j,i); diagonal stays once.
__global__ void k_mirror(int n,int64_t U,const int64_t* Ck,const double* Cv,
                         int64_t* Fk,double* Fv,int* dcnt){
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=U)return;
    int64_t k=Ck[t]; double v=Cv[t]; Fk[t]=k; Fv[t]=v;
    int ii=(int)(k/n), jj=(int)(k%n);
    if(ii<jj){ int pos=atomicAdd(dcnt,1); Fk[pos]=(int64_t)jj*n+ii; Fv[pos]=v; }
}

struct EngResult { float ms; int64_t nnz; double s,as,sq; };

// Run our SpGEMM on device CSR arrays; returns timing + nnz + value checksum.
static EngResult run_ours(const char* task,const int* dOff,const int* dCol,const double* dVal,
                          int n,int64_t nnz,int reps){
    bool AAt=(strcmp(task,"AAt")==0); bool sym=(strcmp(task,"AAtS")==0);
    bool needAt=AAt||sym; int SYM=sym?1:0; int tpb=128;
    int *Boff,*Bcol,*dCursor=nullptr; double *Bval;
    if(needAt){ CK(cudaMalloc(&Boff,(n+1)*sizeof(int))); CK(cudaMalloc(&Bcol,nnz*sizeof(int)));
        CK(cudaMalloc(&Bval,nnz*sizeof(double))); CK(cudaMalloc(&dCursor,n*sizeof(int))); }
    else { Boff=(int*)dOff; Bcol=(int*)dCol; Bval=(double*)dVal; }
    auto build_At=[&](){
        CK(cudaMemset(Boff,0,(n+1)*sizeof(int)));
        k_colcount<<<(int)((nnz+tpb-1)/tpb),tpb>>>(nnz,dCol,Boff+1);
        thrust::device_ptr<int> pb(Boff); thrust::inclusive_scan(pb,pb+n+1,pb);
        CK(cudaMemcpy(dCursor,Boff,n*sizeof(int),cudaMemcpyDeviceToDevice));
        k_scatter<<<(n+tpb-1)/tpb,tpb>>>(n,dOff,dCol,dVal,dCursor,Bcol,Bval);
    };
    int64_t *dFlops,*dGcap,*dHoff; int *dRcap,*dRowcnt,*dCoff;
    CK(cudaMalloc(&dFlops,n*sizeof(int64_t))); CK(cudaMalloc(&dGcap,n*sizeof(int64_t)));
    CK(cudaMalloc(&dHoff,(n+1)*sizeof(int64_t))); CK(cudaMalloc(&dRcap,n*sizeof(int)));
    CK(cudaMalloc(&dRowcnt,n*sizeof(int))); CK(cudaMalloc(&dCoff,(n+1)*sizeof(int)));
    int nblocks=(n+WPB-1)/WPB;
    size_t sh_sym=(size_t)WPB*SH_CAP*sizeof(int);
    size_t sh_num=(size_t)WPB*SH_CAP*(sizeof(int)+sizeof(double));
    if(sh_num>48*1024) CK(cudaFuncSetAttribute(k_numeric,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_num));
    thrust::device_ptr<int64_t> pg(dGcap),ph(dHoff); thrust::device_ptr<int> prc(dRowcnt),pco(dCoff);
    auto structure=[&](int64_t& total_h){
        k_flops<<<(n+tpb-1)/tpb,tpb>>>(n,dOff,dCol,Boff,dFlops);
        k_caps<<<(n+tpb-1)/tpb,tpb>>>(n,dFlops,dRcap,dGcap);
        thrust::exclusive_scan(pg,pg+n,ph);
        int64_t lg,hl; CK(cudaMemcpy(&lg,dGcap+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&hl,dHoff+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost)); total_h=hl+lg;
    };
    if(needAt) build_At();
    int64_t total_h; structure(total_h);
    int* gkeys=nullptr; double* gvals=nullptr;
    if(total_h>0){ CK(cudaMalloc(&gkeys,total_h*sizeof(int))); CK(cudaMalloc(&gvals,total_h*sizeof(double))); CK(cudaMemset(gkeys,0xff,total_h*sizeof(int))); }
    k_symbolic<<<nblocks,WPB*32,sh_sym>>>(n,dOff,dCol,Boff,Bcol,dFlops,dRcap,dHoff,dGcap,gkeys,dRowcnt,SYM);
    thrust::exclusive_scan(prc,prc+n,pco);
    int lastc,cl; CK(cudaMemcpy(&lastc,dRowcnt+n-1,sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&cl,dCoff+n-1,sizeof(int),cudaMemcpyDeviceToHost));
    int64_t Cnnz=(int64_t)cl+lastc;
    int64_t *Ckey,*Ckey_s=nullptr; double *Cval,*Cval_s=nullptr;
    CK(cudaMalloc(&Ckey,Cnnz*sizeof(int64_t))); CK(cudaMalloc(&Cval,Cnnz*sizeof(double)));
    int64_t *Fk=nullptr,*Fks=nullptr; double *Fv=nullptr,*Fvs=nullptr; int* dcnt=nullptr;
    void* d_temp=nullptr; size_t tb=0;
    if(sym){ CK(cudaMalloc(&Fk,2*Cnnz*sizeof(int64_t))); CK(cudaMalloc(&Fks,2*Cnnz*sizeof(int64_t)));
        CK(cudaMalloc(&Fv,2*Cnnz*sizeof(double))); CK(cudaMalloc(&Fvs,2*Cnnz*sizeof(double))); CK(cudaMalloc(&dcnt,sizeof(int)));
        cub::DeviceRadixSort::SortPairs(d_temp,tb,Fk,Fks,Fv,Fvs,(int)(2*Cnnz)); }
    else { CK(cudaMalloc(&Ckey_s,Cnnz*sizeof(int64_t))); CK(cudaMalloc(&Cval_s,Cnnz*sizeof(double)));
        cub::DeviceRadixSort::SortPairs(d_temp,tb,Ckey,Ckey_s,Cval,Cval_s,(int)Cnnz); }
    CK(cudaMalloc(&d_temp,tb));
    int64_t report_nnz=Cnnz;
    auto compute=[&](bool chk,double&s,double&as,double&sq){
        if(needAt) build_At();
        int64_t th; structure(th);
        if(total_h>0) CK(cudaMemset(gkeys,0xff,total_h*sizeof(int)));
        k_symbolic<<<nblocks,WPB*32,sh_sym>>>(n,dOff,dCol,Boff,Bcol,dFlops,dRcap,dHoff,dGcap,gkeys,dRowcnt,SYM);
        thrust::exclusive_scan(prc,prc+n,pco);
        k_numeric<<<nblocks,WPB*32,sh_num>>>(n,dOff,dCol,dVal,Boff,Bcol,Bval,dFlops,dRcap,dHoff,dGcap,gkeys,gvals,dCoff,Ckey,Cval,SYM);
        if(sym){ int u=(int)Cnnz; CK(cudaMemcpy(dcnt,&u,sizeof(int),cudaMemcpyHostToDevice));
            k_mirror<<<(int)((Cnnz+tpb-1)/tpb),tpb>>>(n,Cnnz,Ckey,Cval,Fk,Fv,dcnt);
            int fc; CK(cudaMemcpy(&fc,dcnt,sizeof(int),cudaMemcpyDeviceToHost)); report_nnz=fc;
            cub::DeviceRadixSort::SortPairs(d_temp,tb,Fk,Fks,Fv,Fvs,fc);
            if(chk){ std::vector<double> hv(fc); CK(cudaMemcpy(hv.data(),Fvs,(size_t)fc*sizeof(double),cudaMemcpyDeviceToHost)); checksum(hv,s,as,sq);} }
        else { cub::DeviceRadixSort::SortPairs(d_temp,tb,Ckey,Ckey_s,Cval,Cval_s,(int)Cnnz);
            if(chk){ std::vector<double> hv(Cnnz); CK(cudaMemcpy(hv.data(),Cval_s,Cnnz*sizeof(double),cudaMemcpyDeviceToHost)); checksum(hv,s,as,sq);} }
    };
    double s,as,sq; compute(true,s,as,sq); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); cudaEventRecord(e0);
    for(int r=0;r<reps;r++){double a,b,c; compute(false,a,b,c);}
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=reps;
    EngResult R{ms,report_nnz,s,as,sq};
    cudaFree(dFlops);cudaFree(dGcap);cudaFree(dHoff);cudaFree(dRcap);cudaFree(dRowcnt);cudaFree(dCoff);
    cudaFree(Ckey);cudaFree(Cval); if(Ckey_s)cudaFree(Ckey_s); if(Cval_s)cudaFree(Cval_s);
    if(gkeys)cudaFree(gkeys); if(gvals)cudaFree(gvals); if(d_temp)cudaFree(d_temp);
    if(Fk){cudaFree(Fk);cudaFree(Fks);cudaFree(Fv);cudaFree(Fvs);cudaFree(dcnt);}
    if(needAt){cudaFree(Boff);cudaFree(Bcol);cudaFree(Bval);cudaFree(dCursor);}
    return R;
}
