// Custom hash-based SpGEMM (A*A), fp64 -- v3.
// Fixes v2's two bottlenecks:
//   * thrust::sort_by_key re-allocated temp storage every rep -> use CUB
//     DeviceRadixSort with temp allocated ONCE and reused.
//   * fixed 1024-slot shared hash wasted init on tiny rows -> per-row hash size
//     = next_pow2(~1.33*flops) capped at SH_CAP; init/scan only that many slots.
// Two-tier: shared-memory hash for rows with flops<=SH_LOAD, global-arena hash
// for overflow rows. Canonical CSR via sort of 64-bit (row*n+col) keys.
// Built for sm_80.
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

static const int SH_CAP = 1024;
static const int SH_LOAD = SH_CAP*3/4;
static const int WPB = 4;

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
__host__ __device__ static inline int np2i(int x){ if(x<2)return 2; int p=2; while(p<x)p<<=1; return p; }
__host__ __device__ static inline int64_t np2l(int64_t x){ if(x<2)return 2; int64_t p=2; while(p<x)p<<=1; return p; }

__global__ void k_flops(int n,const int* off,const int* col,int64_t* flops){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int s=off[i],e=off[i+1]; int64_t f=0; for(int p=s;p<e;p++){int k=col[p]; f+=off[k+1]-off[k];} flops[i]=f;
}
// per-row shared cap (0 if overflow) and global cap (0 if shared)
__global__ void k_caps(int n,const int64_t* flops,int* rcap,int64_t* gcap){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int64_t f=flops[i];
    if(f==0){ rcap[i]=0; gcap[i]=0; }
    else if(f<=SH_LOAD){ rcap[i]=np2i((int)((f*4)/3+1)); gcap[i]=0; }
    else { rcap[i]=0; gcap[i]=np2l((f*4)/3+1); }
}
__device__ static inline int ins(int* keys,int cap,int key){          // works for shared or global slice
    unsigned h=((unsigned)key*2654435761u)&(cap-1);
    while(true){int cur=keys[h]; if(cur==key)return (int)h;
        if(cur==-1){int p=atomicCAS(&keys[h],-1,key); if(p==-1||p==key)return (int)h;} h=(h+1)&(cap-1);}
}

__global__ void k_symbolic(int n,const int* Aoff,const int* Acol,const int64_t* flops,
                           const int* rcap,const int64_t* hoff,const int64_t* gcap,int* gkeys,int* rowcnt){
    extern __shared__ int sh[];
    int wib=threadIdx.x>>5, lane=threadIdx.x&31;
    int i=(blockIdx.x*WPB)+wib; if(i>=n)return;
    int64_t f=flops[i]; if(f==0){ if(lane==0)rowcnt[i]=0; return; }
    int s=Aoff[i],e=Aoff[i+1];
    int rc=rcap[i];
    if(rc>0){
        int* keys=sh+wib*SH_CAP;
        for(int t=lane;t<rc;t+=32)keys[t]=-1; __syncwarp();
        for(int p=s;p<e;p++){int k=Acol[p]; int ks=Aoff[k],ke=Aoff[k+1];
            for(int q=ks+lane;q<ke;q+=32) ins(keys,rc,Acol[q]); }
        __syncwarp();
        int cnt=0; for(int t=lane;t<rc;t+=32) if(keys[t]!=-1)cnt++;
        for(int o=16;o>0;o>>=1)cnt+=__shfl_down_sync(0xffffffff,cnt,o);
        if(lane==0)rowcnt[i]=cnt;
    } else {
        int c=(int)gcap[i]; int64_t base=hoff[i]; int* keys=gkeys+base;
        for(int p=s;p<e;p++){int k=Acol[p]; int ks=Aoff[k],ke=Aoff[k+1];
            for(int q=ks+lane;q<ke;q+=32) ins(keys,c,Acol[q]); }
        __syncwarp();
        int cnt=0; for(int t=lane;t<c;t+=32) if(keys[t]!=-1)cnt++;
        for(int o=16;o>0;o>>=1)cnt+=__shfl_down_sync(0xffffffff,cnt,o);
        if(lane==0)rowcnt[i]=cnt;
    }
}
// numeric: accumulate, emit 64-bit key (row*n+col) + value (unsorted)
__global__ void k_numeric(int n,const int* Aoff,const int* Acol,const double* Aval,
                          const int64_t* flops,const int* rcap,const int64_t* hoff,const int64_t* gcap,
                          int* gkeys,double* gvals,const int* Coff,int64_t* Ckey,double* Cval){
    extern __shared__ char shm[];
    int wib=threadIdx.x>>5, lane=threadIdx.x&31;
    int* skeys=(int*)shm + wib*SH_CAP;
    double* svals=(double*)((int*)shm + WPB*SH_CAP) + wib*SH_CAP;
    int i=(blockIdx.x*WPB)+wib; if(i>=n)return;
    int64_t f=flops[i]; if(f==0)return;
    int s=Aoff[i],e=Aoff[i+1]; int wpos=Coff[i]; int rc=rcap[i];
    if(rc>0){
        for(int t=lane;t<rc;t+=32){skeys[t]=-1; svals[t]=0.0;} __syncwarp();
        for(int p=s;p<e;p++){int k=Acol[p]; double aik=Aval[p]; int ks=Aoff[k],ke=Aoff[k+1];
            for(int q=ks+lane;q<ke;q+=32){int slot=ins(skeys,rc,Acol[q]); atomicAdd(&svals[slot],aik*Aval[q]); } }
        __syncwarp();
        for(int t=lane;t<rc;t+=32){int key=skeys[t]; unsigned mk=__ballot_sync(0xffffffff,key!=-1);
            int rank=__popc(mk&((1u<<lane)-1));
            if(key!=-1){int pos=wpos+rank; Ckey[pos]=(int64_t)i*n+key; Cval[pos]=svals[t];}
            wpos+=__popc(mk); }
    } else {
        int c=(int)gcap[i]; int64_t base=hoff[i]; int* keys=gkeys+base; double* vals=gvals+base;
        for(int p=s;p<e;p++){int k=Acol[p]; double aik=Aval[p]; int ks=Aoff[k],ke=Aoff[k+1];
            for(int q=ks+lane;q<ke;q+=32){int slot=ins(keys,c,Acol[q]); atomicAdd(&vals[slot],aik*Aval[q]); } }
        __syncwarp();
        for(int t=lane;t<c;t+=32){int key=keys[t]; unsigned mk=__ballot_sync(0xffffffff,key!=-1);
            int rank=__popc(mk&((1u<<lane)-1));
            if(key!=-1){int pos=wpos+rank; Ckey[pos]=(int64_t)i*n+key; Cval[pos]=vals[t];}
            wpos+=__popc(mk); }
    }
}
static void checksum(const std::vector<double>& v,double&s,double&as,double&sq){
    s=as=sq=0; for(double x:v){s+=x;as+=x<0?-x:x;sq+=x*x;}
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <tag> <bin.csr> [reps]\n",argv[0]);return 1;}
    const char* tag=argv[1]; int reps=argc>3?atoi(argv[3]):10;
    Csr h=load_bin(argv[2]); int n=h.n; int64_t nnz=h.nnz;
    int *dOff,*dCol; double *dVal;
    CK(cudaMalloc(&dOff,(n+1)*sizeof(int))); CK(cudaMalloc(&dCol,nnz*sizeof(int))); CK(cudaMalloc(&dVal,nnz*sizeof(double)));
    CK(cudaMemcpy(dOff,h.indptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCol,h.indices.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVal,h.data.data(),nnz*sizeof(double),cudaMemcpyHostToDevice));

    int64_t *dFlops,*dGcap,*dHoff; int *dRcap,*dRowcnt,*dCoff;
    CK(cudaMalloc(&dFlops,n*sizeof(int64_t))); CK(cudaMalloc(&dGcap,n*sizeof(int64_t)));
    CK(cudaMalloc(&dHoff,(n+1)*sizeof(int64_t))); CK(cudaMalloc(&dRcap,n*sizeof(int)));
    CK(cudaMalloc(&dRowcnt,n*sizeof(int))); CK(cudaMalloc(&dCoff,(n+1)*sizeof(int)));
    int tpb=128;
    int nblocks=(n+WPB-1)/WPB;
    size_t sh_sym=(size_t)WPB*SH_CAP*sizeof(int);
    size_t sh_num=(size_t)WPB*SH_CAP*(sizeof(int)+sizeof(double));
    if(sh_num>48*1024) CK(cudaFuncSetAttribute(k_numeric,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_num));
    thrust::device_ptr<int64_t> pg(dGcap),ph(dHoff);
    thrust::device_ptr<int> prc(dRowcnt),pco(dCoff);
    // ---- one-time SIZING pass (deterministic): total global-arena + Cnnz ----
    k_flops<<<(n+tpb-1)/tpb,tpb>>>(n,dOff,dCol,dFlops);
    k_caps<<<(n+tpb-1)/tpb,tpb>>>(n,dFlops,dRcap,dGcap);
    thrust::exclusive_scan(pg,pg+n,ph);
    int64_t last_g,hoff_last; CK(cudaMemcpy(&last_g,dGcap+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hoff_last,dHoff+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost));
    int64_t total_h=hoff_last+last_g;
    int* gkeys=nullptr; double* gvals=nullptr;
    if(total_h>0){ CK(cudaMalloc(&gkeys,total_h*sizeof(int))); CK(cudaMalloc(&gvals,total_h*sizeof(double))); CK(cudaMemset(gkeys,0xff,total_h*sizeof(int))); }
    k_symbolic<<<nblocks,WPB*32,sh_sym>>>(n,dOff,dCol,dFlops,dRcap,dHoff,dGcap,gkeys,dRowcnt);
    thrust::exclusive_scan(prc,prc+n,pco);
    int lastc,coff_last; CK(cudaMemcpy(&lastc,dRowcnt+n-1,sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&coff_last,dCoff+n-1,sizeof(int),cudaMemcpyDeviceToHost));
    int64_t Cnnz=(int64_t)coff_last+lastc;
    int64_t *Ckey,*Ckey_s; double *Cval,*Cval_s;
    CK(cudaMalloc(&Ckey,Cnnz*sizeof(int64_t))); CK(cudaMalloc(&Ckey_s,Cnnz*sizeof(int64_t)));
    CK(cudaMalloc(&Cval,Cnnz*sizeof(double)));  CK(cudaMalloc(&Cval_s,Cnnz*sizeof(double)));
    void* d_temp=nullptr; size_t temp_bytes=0;
    cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,Ckey,Ckey_s,Cval,Cval_s,(int)Cnnz);
    CK(cudaMalloc(&d_temp,temp_bytes));

    // Timed compute() runs the FULL pipeline (symbolic + numeric + sort) each
    // call, reusing preallocated buffers -- fair vs cuSPARSE's full SpGEMM.
    auto compute=[&](bool chk,double&s,double&as,double&sq){
        k_flops<<<(n+tpb-1)/tpb,tpb>>>(n,dOff,dCol,dFlops);
        k_caps<<<(n+tpb-1)/tpb,tpb>>>(n,dFlops,dRcap,dGcap);
        thrust::exclusive_scan(pg,pg+n,ph);
        if(total_h>0){ CK(cudaMemset(gkeys,0xff,total_h*sizeof(int))); }
        k_symbolic<<<nblocks,WPB*32,sh_sym>>>(n,dOff,dCol,dFlops,dRcap,dHoff,dGcap,gkeys,dRowcnt);
        thrust::exclusive_scan(prc,prc+n,pco);
        if(total_h>0){ CK(cudaMemset(gvals,0,total_h*sizeof(double))); }
        k_numeric<<<nblocks,WPB*32,sh_num>>>(n,dOff,dCol,dVal,dFlops,dRcap,dHoff,dGcap,gkeys,gvals,dCoff,Ckey,Cval);
        cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,Ckey,Ckey_s,Cval,Cval_s,(int)Cnnz);
        if(chk){ std::vector<double> hv(Cnnz); CK(cudaMemcpy(hv.data(),Cval_s,Cnnz*sizeof(double),cudaMemcpyDeviceToHost)); checksum(hv,s,as,sq); }
    };
    double s,as,sq; compute(true,s,as,sq); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int r=0;r<reps;r++){double a,b,c; compute(false,a,b,c);}
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=reps;
    printf("  __chk s=%.6e as=%.6e sq=%.6e\n",s,as,sq);
    printf("RESULT,%s,AA,%d,%lld,%lld,%.4f\n",tag,n,(long long)nnz,(long long)Cnnz,ms);
    return 0;
}
