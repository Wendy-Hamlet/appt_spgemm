// Custom hash-based SpGEMM (A*A), fp64, one warp per output row.
// V1: correctness-first. Global-memory open-addressing hash per row (sized from
// the per-row flop upper bound), single hashing pass that accumulates values,
// then distinct-count -> C structure -> compact -> global sort to canonical CSR.
// This is the "global-hash" strategy branch of the eventual dispatcher.
//
// Built for sm_80 so it ports to A100. Output mirrors the cuSPARSE baseline:
//   RESULT,<tag>,AA,n,nnz,C_nnz,ms   and a "  __chk s=.. as=.. sq=.." line.
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

struct Csr { int n; int64_t nnz; std::vector<int> indptr, indices; std::vector<double> data; };

static Csr load_bin(const char* path){
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(1);}
    char magic[4]; if(fread(magic,1,4,f)!=4||memcmp(magic,"CSR1",4)){fprintf(stderr,"bad magic\n");exit(1);}
    Csr c; int pad; if(fread(&c.n,4,1,f)!=1)exit(1); if(fread(&pad,4,1,f)!=1)exit(1);
    if(fread(&c.nnz,8,1,f)!=1)exit(1);
    c.indptr.resize(c.n+1); c.indices.resize(c.nnz); c.data.resize(c.nnz);
    if((int64_t)fread(c.indptr.data(),4,c.n+1,f)!=c.n+1)exit(1);
    if((int64_t)fread(c.indices.data(),4,c.nnz,f)!=c.nnz)exit(1);
    if((int64_t)fread(c.data.data(),8,c.nnz,f)!=c.nnz)exit(1);
    fclose(f); return c;
}

__host__ __device__ static inline int64_t next_pow2_ll(int64_t x){
    if(x<2) return 2; int64_t p=2; while(p<x) p<<=1; return p;
}

// per-row flop upper bound = sum_{k in row i} deg(k)
__global__ void k_flops(int n,const int* off,const int* col,int64_t* flops){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int s=off[i],e=off[i+1]; int64_t f=0;
    for(int p=s;p<e;p++){ int k=col[p]; f += off[k+1]-off[k]; }
    flops[i]=f;
}
// per-row hash capacity (power of 2, ~1.33x flops), and its exclusive offset base
__global__ void k_cap(int n,const int64_t* flops,int64_t* cap){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int64_t f=flops[i]; cap[i]= f==0?0:next_pow2_ll((f*4)/3+1);
}

__device__ static inline int find_or_insert(int* keys,int64_t base,int cap,int key){
    unsigned h=((unsigned)key*2654435761u)&(cap-1);
    while(true){
        int cur=keys[base+h];
        if(cur==key) return (int)h;
        if(cur==-1){ int prev=atomicCAS(&keys[base+h],-1,key); if(prev==-1||prev==key) return (int)h; }
        h=(h+1)&(cap-1);
    }
}

// one warp per row: accumulate all products into the row's hash region
__global__ void k_build(int n,const int* Aoff,const int* Acol,const double* Aval,
                        const int64_t* hoff,const int64_t* cap,int* keys,double* vals){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int lane=threadIdx.x&31;
    if(warp>=n) return;
    int i=warp; int c=(int)cap[i]; if(c==0) return;
    int64_t base=hoff[i];
    int s=Aoff[i],e=Aoff[i+1];
    for(int p=s;p<e;p++){
        int k=Acol[p]; double aik=Aval[p];
        int ks=Aoff[k],ke=Aoff[k+1];
        for(int q=ks+lane;q<ke;q+=32){
            int j=Acol[q]; double v=aik*Aval[q];
            int slot=find_or_insert(keys,base,c,j);
            atomicAdd(&vals[base+slot],v);
        }
    }
}
// distinct count per row = non-empty slots
__global__ void k_count(int n,const int64_t* hoff,const int64_t* cap,const int* keys,int* rowcnt){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;
    if(warp>=n) return; int i=warp; int c=(int)cap[i]; int64_t base=hoff[i];
    int cnt=0; for(int s=lane;s<c;s+=32) if(keys[base+s]!=-1) cnt++;
    for(int o=16;o>0;o>>=1) cnt+=__shfl_down_sync(0xffffffff,cnt,o);
    if(lane==0) rowcnt[i]=cnt;
}
// compact non-empty slots into C segment [Coff[i], Coff[i+1])
__global__ void k_extract(int n,const int64_t* hoff,const int64_t* cap,const int* keys,
                          const double* vals,const int* Coff,int* Ccol,double* Cval,int* Crow){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;
    if(warp>=n) return; int i=warp; int c=(int)cap[i]; int64_t base=hoff[i];
    int wpos=Coff[i];               // warp-shared write cursor via atomics on shared? use ballot compaction
    for(int s=lane;s<c;s+=32){
        int key=keys[base+s]; unsigned mask=__ballot_sync(0xffffffff,key!=-1);
        int rank=__popc(mask&((1u<<lane)-1));
        if(key!=-1){ int pos=wpos+rank; Ccol[pos]=key; Cval[pos]=vals[base+s]; Crow[pos]=i; }
        wpos+=__popc(mask);
    }
}

static void checksum(const std::vector<double>& v,double&s,double&as,double&sq){
    s=as=sq=0; for(double x:v){ s+=x; as+=x<0?-x:x; sq+=x*x; }
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

    int64_t *dFlops,*dCap,*dHoff; CK(cudaMalloc(&dFlops,n*sizeof(int64_t)));
    CK(cudaMalloc(&dCap,n*sizeof(int64_t))); CK(cudaMalloc(&dHoff,(n+1)*sizeof(int64_t)));
    int *dRowcnt,*dCoff; CK(cudaMalloc(&dRowcnt,n*sizeof(int))); CK(cudaMalloc(&dCoff,(n+1)*sizeof(int)));

    auto run=[&](bool measure,double& s,double& as,double& sq,int64_t& Cnnz)->void{
        int tpb=128;
        k_flops<<<(n+tpb-1)/tpb,tpb>>>(n,dOff,dCol,dFlops);
        k_cap<<<(n+tpb-1)/tpb,tpb>>>(n,dFlops,dCap);
        // exclusive scan cap -> hoff
        thrust::device_ptr<int64_t> pcap(dCap),phoff(dHoff);
        thrust::exclusive_scan(pcap,pcap+n,phoff);
        int64_t last_cap; CK(cudaMemcpy(&last_cap,dCap+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost));
        int64_t hoff_last; CK(cudaMemcpy(&hoff_last,dHoff+n-1,sizeof(int64_t),cudaMemcpyDeviceToHost));
        int64_t total_h=hoff_last+last_cap;
        CK(cudaMemcpy(dHoff+n,&total_h,sizeof(int64_t),cudaMemcpyHostToDevice));
        int *keys; double *vals;
        CK(cudaMalloc(&keys,total_h*sizeof(int))); CK(cudaMalloc(&vals,total_h*sizeof(double)));
        CK(cudaMemset(keys,0xff,total_h*sizeof(int)));   // -1
        CK(cudaMemset(vals,0,total_h*sizeof(double)));
        int warps_per_block=4, wtpb=warps_per_block*32;
        int nblocks=(n+warps_per_block-1)/warps_per_block;
        k_build<<<nblocks,wtpb>>>(n,dOff,dCol,dVal,dHoff,dCap,keys,vals);
        k_count<<<nblocks,wtpb>>>(n,dHoff,dCap,keys,dRowcnt);
        thrust::device_ptr<int> prc(dRowcnt),pco(dCoff);
        thrust::exclusive_scan(prc,prc+n,pco);
        int lastc; CK(cudaMemcpy(&lastc,dRowcnt+n-1,sizeof(int),cudaMemcpyDeviceToHost));
        int coff_last; CK(cudaMemcpy(&coff_last,dCoff+n-1,sizeof(int),cudaMemcpyDeviceToHost));
        Cnnz=(int64_t)coff_last+lastc;
        int *Ccol,*Crow; double *Cval;
        CK(cudaMalloc(&Ccol,Cnnz*sizeof(int))); CK(cudaMalloc(&Crow,Cnnz*sizeof(int))); CK(cudaMalloc(&Cval,Cnnz*sizeof(double)));
        k_extract<<<nblocks,wtpb>>>(n,dHoff,dCap,keys,vals,dCoff,Ccol,Cval,Crow);
        // canonical sort by (row,col): key = row*(int64)n + col
        int64_t* k64; CK(cudaMalloc(&k64,Cnnz*sizeof(int64_t)));
        {
            thrust::device_ptr<int> pcol(Ccol),prow(Crow);
            thrust::device_ptr<int64_t> pk(k64);
            thrust::transform(prow,prow+Cnnz,pcol,pk,
                [n]__device__(int r,int c){return (int64_t)r*n+c;});
            // sort col & val by key
            thrust::device_ptr<double> pval(Cval);
            auto zip=thrust::make_zip_iterator(thrust::make_tuple(pcol,pval));
            thrust::sort_by_key(pk,pk+Cnnz,zip);
        }
        if(!measure){
            std::vector<double> hv(Cnnz); CK(cudaMemcpy(hv.data(),Cval,Cnnz*sizeof(double),cudaMemcpyDeviceToHost));
            checksum(hv,s,as,sq);
        }
        cudaFree(keys);cudaFree(vals);cudaFree(Ccol);cudaFree(Crow);cudaFree(Cval);cudaFree(k64);
    };

    double s,as,sq; int64_t Cnnz=0;
    run(false,s,as,sq,Cnnz);                 // correctness pass
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int r=0;r<reps;r++){ double a,b,c; int64_t nn; run(true,a,b,c,nn); }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=reps;
    printf("  __chk s=%.6e as=%.6e sq=%.6e\n",s,as,sq);
    printf("RESULT,%s,AA,%d,%lld,%lld,%.4f\n",tag,n,(long long)nnz,(long long)Cnnz,ms);
    return 0;
}
