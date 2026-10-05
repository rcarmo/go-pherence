// Offline PTX generator source, compiled without a CUDA context/device.
// Two-pass global max/fixed-scale compensated reduction and clipping.
extern "C" __global__ void training_global_norm(const float*a,const float*b,float*out,unsigned n,unsigned mode,float previousCount,float maxNorm){
 if(blockIdx.x*blockDim.x+threadIdx.x!=0)return;
 float maximum=a[0],sum=a[1],invalid=a[2],correction=a[3];
 if(!isfinite(maximum)||!isfinite(sum)||!isfinite(invalid)||!isfinite(correction)||invalid!=0.0f||maximum<0||sum<0||(maximum==0&&sum!=0)||sum>previousCount*(1.0f+0.000002f))invalid=1.0f;
 if(mode==1){if(maximum>0&&sum<1)invalid=1;float scale=1.0f;if(invalid==0&&maximum>0){float root=sqrtf(sum),ratio=maxNorm/maximum;if(ratio<root)scale=ratio/root;}if(invalid!=0)scale=0;out[0]=maximum;out[1]=sum;out[2]=invalid;out[3]=scale;return;}
 unsigned chunks=(n-1)/256+1;
 for(unsigned j=0;j<chunks;j++){float m=b[3*j],s=b[3*j+1],flag=b[3*j+2];unsigned count=n-j*256;if(count>256)count=256;if(!isfinite(m)||!isfinite(s)||!isfinite(flag)||flag!=0||m<0||s<0||s>(float)count||(m==0&&s!=0)||(m>0&&s<1))invalid=1.0f;if(mode==0){if(m>maximum)maximum=m;}else{if(m>maximum)invalid=1;}}
 float total=sum,comp=correction;if(mode==0){total=0;comp=0;}else if(invalid==0&&maximum>0){for(unsigned j=0;j<chunks;j++){float r=b[3*j]/maximum,term=(r*r)*b[3*j+1]-comp,next=total+term;comp=(next-total)-term;total=next;}}
 out[0]=maximum;out[1]=total;out[2]=invalid;out[3]=comp;
}
// Normalized one-row LSD square/loss/seeds, compensated square reduction.
extern "C" __global__ void training_lsd_loss(const float*p,const float*d,const float*logvar,float*out,unsigned channels,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==2){if(i!=0)return;float invalid=0.0f,cutoff=logvar[0];for(unsigned j=0;j<channels;j++){float pv=p[j],yv=d[j],diff=pv-yv;if(!(fabsf(pv)<=3.402823466e38f)||!(fabsf(yv)<=3.402823466e38f)||!(fabsf(diff)<=cutoff))invalid=1.0f;}out[0]=invalid;return;}
 float scale=expf(logvar[0])/(float)channels;
 if(mode==1){if(i>=channels)return;float difference=p[i]-d[i];out[i]=(2.0f*difference)*scale;return;}
 if(i!=0)return;float square=0.0f,correction=0.0f;for(unsigned j=0;j<channels;j++){float diff=p[j]-d[j],term=diff*diff-correction,next=square+term;correction=(next-square)-term;square=next;}float normalized=square*scale;out[0]=square;out[1]=normalized-logvar[0];out[2]=normalized-1.0f;
}
// Stable BCE and weighted mean gradient; compensated scalar loss reduction.
__device__ __forceinline__ unsigned eos_admission_units(float x,float y,unsigned remaining,unsigned budget){
 unsigned xb=__float_as_uint(x)&0x7fffffffu,yb=__float_as_uint(y)&0x7fffffffu;
 unsigned xe=xb>>23,ye=yb>>23;
 if(xe==255u||ye==255u)return budget+1u;
 unsigned xm=xb&0x7fffffu,ym=yb&0x7fffffu;
 if(xe==0u)xe=1u;else xm|=0x800000u;
 if(ye==0u)ye=1u;else ym|=0x800000u;
 xm=(xm+4095u)>>12;ym=(ym+4095u)>>12;
 unsigned p=xm*ym;if(p==0u)return 0u;
 unsigned sum=xe+ye;
 if(sum>=371u){unsigned shift=sum-371u;if(shift>=31u)return budget+1u;if(p>(remaining>>shift))return budget+1u;return p<<shift;}
 unsigned shift=371u-sum;
 if(shift>=32u){if(remaining==0u)return budget+1u;return 1u;}
 unsigned value=p>>shift,mask=0xffffffffu>>(32u-shift);if((p&mask)!=0u)value++;
 return value;
}
extern "C" __global__ void training_eos_admission(const float*hidden,const float*weight,const float*bias,float*out,unsigned rows,unsigned width,unsigned budget){
 if(blockIdx.x*blockDim.x+threadIdx.x!=0)return;
 for(unsigned row=0;row<rows;row++){
  unsigned used=eos_admission_units(bias[0],1.f,budget,budget);if(used>budget){out[0]=1.f;return;}
  unsigned remaining=budget-used;
  for(unsigned col=0;col<width;col++){used=eos_admission_units(hidden[row*width+col],weight[col],remaining,budget);if(used>remaining){out[0]=1.f;return;}remaining-=used;}
 }
 out[0]=0.f;
}
extern "C" __global__ void training_eos(const float*x,const float*labels,float*out,unsigned n,unsigned active,unsigned mode,float weight){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==2){if(i!=0)return;float sum=0.0f,correction=0.0f;for(unsigned j=0;j<n;j++)if(labels[j]>=0.0f){float term=x[j]-correction,next=sum+term;correction=(next-sum)-term;sum=next;}out[0]=sum/(float)active;return;}
 if(i>=n)return;if(labels[i]<0.0f){out[i]=0.0f;return;}float v=x[i],e=expf(-fabsf(v));
 if(mode==0){float base=labels[i]==1.0f?(v<0.0f?-v:0.0f):(v>0.0f?v:0.0f);out[i]=base+log1pf(e);}
 else{float residual=labels[i]==1.0f?(v>=0.0f?-e/(1.0f+e):-1.0f/(1.0f+e)):(v>=0.0f?1.0f/(1.0f+e):e/(1.0f+e));float gradient=residual/(float)active;out[i]=gradient*weight;}
}
// Sinusoidal value/JVP and reverse-over-forward time adjoint, cos then sin.
extern "C" __global__ void training_time_embedding(const float*f,const float*time,const float*a,const float*b,float*out,unsigned half,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode<2){if(i>=2*half)return;unsigned j=i%half;float freq=f[j],angle=time[0]*freq,s=sinf(angle),c=cosf(angle);if(mode==0)out[i]=i<half?c:s;else{float derivative=i<half?-s*freq:c*freq;out[i]=derivative*time[1];}}
 else{if(i!=0)return;float sum=0.0f;for(unsigned j=0;j<half;j++){float freq=f[j],angle=time[0]*freq,s=sinf(angle),c=cosf(angle);float fc=-s*freq,fs=c*freq,sc=-c*freq*freq,ss=-s*freq*freq;sum+=a[j]*fc+a[half+j]*fs;sum+=b[j]*sc*time[1]+b[half+j]*ss*time[1];}out[0]=sum;}
}
// Disjoint gather and sequence-order repeated-token VJP; no atomics.
// Mode2 uses admitted occurrence offsets + row indices, retaining dense output.
// Mode3 uses active-token slots + offsets + row indices, padding output to rows.
extern "C" __global__ void training_embedding(const float*x,const float*ids,float*out,unsigned rows,unsigned width,unsigned vocabulary,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==0){if(i>=rows*width)return;float id=ids[i/width];if(!(id>=0.0f&&id<(float)vocabulary)){out[i]=0.0f;return;}unsigned token=(unsigned)id;out[i]=id==(float)token?x[token*width+i%width]:0.0f;}
 else if(mode==3){if(i>=rows*width)return;unsigned slot=i/width,col=i%width;float token=ids[slot],lo=ids[rows+slot],hi=ids[rows+slot+1],sum=0.0f;if(!(token>=0.0f&&token<(float)vocabulary)||token!=(float)(unsigned)token){out[i]=0.0f;return;}if(!(lo>=0.0f&&hi>=lo&&hi<=(float)rows)||lo!=(float)(unsigned)lo||hi!=(float)(unsigned)hi){out[i]=0.0f;return;}for(unsigned at=(unsigned)lo;at<(unsigned)hi;at++){float row=ids[2*rows+1+at];if(row>=0.0f&&row<(float)rows&&row==(float)(unsigned)row)sum+=x[(unsigned)row*width+col];}out[i]=sum;}
 else if(mode==2){if(i>=vocabulary*width)return;unsigned token=i/width,col=i%width;float lo=ids[token],hi=ids[token+1],sum=0.0f;if(!(lo>=0.0f&&hi>=lo&&hi<=(float)rows)||lo!=(float)(unsigned)lo||hi!=(float)(unsigned)hi){out[i]=0.0f;return;}for(unsigned at=(unsigned)lo;at<(unsigned)hi;at++){float row=ids[vocabulary+1+at];if(row>=0.0f&&row<(float)rows&&row==(float)(unsigned)row)sum+=x[(unsigned)row*width+col];}out[i]=sum;}
 else{if(i>=vocabulary*width)return;unsigned token=i/width,col=i%width;float sum=0.0f;for(unsigned row=0;row<rows;row++)if(ids[row]==(float)token)sum+=x[row*width+col];out[i]=sum;}
}
// Compact VJP accumulation into dense ping-pong resident storage. Sorted token
// slots allow logarithmic lookup; padding sorts after every admitted token.
extern "C" __global__ void training_embedding_accumulate(const float*previous,const float*compact,const float*index,float*out,unsigned rows,unsigned width,unsigned vocabulary,float scale){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=vocabulary*width)return;
 unsigned token=i/width,lo=0,hi=rows;
 while(lo<hi){unsigned mid=lo+(hi-lo)/2;float id=index[mid];if(id>=0.0f&&id<(float)token)lo=mid+1;else hi=mid;}
 float contribution=0.0f;if(lo<rows&&index[lo]==(float)token)contribution=compact[lo*width+i%width];
 out[i]=previous[i]+scale*contribution;
}
extern "C" __global__ void training_relu(const float* x,const float* seed,float* out,unsigned n,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=mode==0?(x[i]<0?0:x[i]):(x[i]>0?seed[i]:0);
}

extern "C" __global__ void training_norm_stats(const float* x,float* out,unsigned n,unsigned nonnegative){
 unsigned chunk=blockIdx.x*blockDim.x+threadIdx.x,start=chunk*256;if(start>=n)return;
 unsigned end=start+256;if(end>n)end=n;float maximum=0,invalid=0;
 for(unsigned i=start;i<end;i++){float v=x[i];if(!(fabsf(v)<=3.402823466e38f)||(nonnegative&&v<0))invalid=1;else if(fabsf(v)>maximum)maximum=fabsf(v);}
 float sum=0,correction=0;if(maximum>0){for(unsigned i=start;i<end;i++){float scaled=x[i]/maximum,term=scaled*scaled-correction,next=sum+term;correction=(next-sum)-term;sum=next;}}
 out[chunk*3]=maximum;out[chunk*3+1]=sum;out[chunk*3+2]=invalid;
}

extern "C" __global__ void training_gelu(const float* x,const float* seed,float* out,unsigned n,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=x[i];
 float u=0.7978845608028654f*(v+((0.044715f*v)*v)*v);float e=expf(-2*fabsf(u));
 float halfOnePlus=u<0?e/(1+e):1/(1+e);float sech2=(4*e)/((1+e)*(1+e));
 if(mode==0)out[i]=v*halfOnePlus;
 else{float du=0.7978845608028654f*(1+(0.134145f*v)*v);float derivative=halfOnePlus+((.5f*v)*sech2)*du;out[i]=seed[i]*derivative;}
}
extern "C" __global__ void training_row(const float* x,const float* b,float* out,unsigned rows,unsigned width,unsigned parts,unsigned slot,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==0){if(i>=rows*width)return;out[i]=x[(i/width*parts+slot)*width+i%width];}
 else if(mode==1){if(i>=rows*parts*width)return;unsigned col=i%width,part=(i/width)%parts,row=i/(parts*width);out[i]=part==slot?x[row*width+col]:0;}
 else if(mode==2){if(i>=rows*width)return;out[i]=x[i]*b[i%width];}
 else{if(i>=width)return;float sum=0;for(unsigned row=0;row<rows;row++)sum+=x[row*width+i]*b[row*width+i];out[i]=sum;}
}

__device__ bool training_key_allowed(unsigned query,unsigned key,unsigned context){return key<=query&&(context==0||query-key<context);}
extern "C" __global__ void training_attention_forward(const float* q,const float* k,const float* v,const float* aux,float* out,unsigned rows,unsigned heads,unsigned dim,unsigned context,unsigned mode,float scale){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==0){if(i>=heads*rows*rows)return;unsigned key=i%rows,query=(i/rows)%rows,head=i/(rows*rows);
  if(!training_key_allowed(query,key,context)){out[i]=0;return;}float sum=0;for(unsigned d=0;d<dim;d++)sum+=q[(query*heads+head)*dim+d]*k[(key*heads+head)*dim+d];out[i]=sum*scale;
 }else if(mode==1){if(i>=heads*rows)return;unsigned query=i%rows,base=i*rows;float maximum=-3.402823466e38f;
  for(unsigned key=0;key<rows;key++)if(training_key_allowed(query,key,context)&&aux[base+key]>maximum)maximum=aux[base+key];
  float sum=0;for(unsigned key=0;key<rows;key++)if(training_key_allowed(query,key,context))sum+=expf(aux[base+key]-maximum);
  for(unsigned key=0;key<rows;key++)out[base+key]=training_key_allowed(query,key,context)?expf(aux[base+key]-maximum)/sum:0;
 }else{if(i>=rows*heads*dim)return;unsigned d=i%dim,head=(i/dim)%heads,query=i/(heads*dim);float sum=0;
  for(unsigned key=0;key<rows;key++)if(training_key_allowed(query,key,context))sum+=aux[(head*rows+query)*rows+key]*v[(key*heads+head)*dim+d];out[i]=sum;
 }
}

extern "C" __global__ void training_rope(const float* x,const float* table,float* out,unsigned rows,unsigned heads,unsigned dim,unsigned mode){
 unsigned pair=blockIdx.x*blockDim.x+threadIdx.x;if(pair>=rows*heads*(dim/2))return;
 unsigned i=2*pair,row=i/(heads*dim),col=i%dim;
 float c=table[row*dim+col],s=table[row*dim+col+1];if(mode==1)s=-s;
 out[i]=x[i]*c-x[i+1]*s;out[i+1]=x[i+1]*c+x[i]*s;
}

extern "C" __global__ void training_silu_dual_adjoint(const float* x,const float* xt,const float* gv,const float* gt,float* out,unsigned n,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float e=expf(-fabsf(x[i]));float s=x[i]>=0?1/(1+e):e/(1+e);
 float first=s*(1+x[i]*(1-s)),second=s*(1-s)*(2+x[i]*(1-2*s));
 out[i]=mode==0?gv[i]*first+(gt[i]*second)*xt[i]:gt[i]*first;
}

extern "C" __global__ void training_vector(const float* a,const float* b,float* out,unsigned n,unsigned sourceCount,unsigned offset,unsigned mode,float scale){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(mode==0)out[i]=a[i]+b[i]*scale;
 else if(mode==1)out[i]=a[offset+i];
 else out[i]=(i>=offset&&i-offset<sourceCount)?a[i-offset]:0;
}

extern "C" __global__ void training_linear_dual_forward(
 const float* x,const float* xt,const float* w,const float* bias,float* out,
 unsigned rows,unsigned inputDim,unsigned outputDim,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*outputDim)return;
 unsigned row=i/outputDim,col=i%outputDim;float sum=0;
 for(unsigned q=0;q<inputDim;q++)sum+=(mode==0?x[row*inputDim+q]:xt[row*inputDim+q])*w[col*inputDim+q];
 if(mode==0)sum+=bias[col];out[i]=sum;
}
extern "C" __global__ void training_silu_dual_forward(const float* x,const float* xt,float* out,unsigned n,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float e=expf(-fabsf(x[i]));float s=x[i]>=0?1/(1+e):e/(1+e);
 out[i]=mode==0?x[i]*s:(xt[i]*s)*(1+x[i]*(1-s));
}
extern "C" __global__ void training_mod_dual_forward(const float* a,const float* at,const float* b,const float* bt,float* out,unsigned n,unsigned kind,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(kind==0){float scale=1+b[n+i];out[i]=mode==0?a[i]*scale+b[i]:(at[i]*scale+a[i]*bt[n+i])+bt[i];}
 else out[i]=mode==0?a[i]*b[i]:at[i]*b[i]+a[i]*bt[i];
}

extern "C" __global__ void training_norm_dual_forward(
 const float* x,const float* xt,const float* w,const float* bias,const float* stats,float* out,
 unsigned rows,unsigned width,unsigned kind,unsigned mode,float eps){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode==0){
  if(i>=rows)return;unsigned base=i*width;float mean=0,dmean=0;
  for(unsigned c=0;c<width;c++){mean+=x[base+c];dmean+=xt[base+c];}mean/=float(width);dmean/=float(width);
  float variance=0,dvar=0;for(unsigned c=0;c<width;c++){float center=x[base+c]-mean,dc=xt[base+c]-dmean;variance+=center*center;dvar+=2*center*dc;}
  float denom=kind==0?float(width):float(width-1);variance/=denom;dvar/=denom;
  float inv=1.0f/sqrtf(variance+eps);float dinv=-.5f*inv*inv*inv*dvar;
  out[i*5]=mean;out[i*5+1]=dmean;out[i*5+2]=inv;out[i*5+3]=dinv;out[i*5+4]=dvar;
 }else{
  if(i>=rows*width)return;unsigned row=i/width,col=i%width;
  float value=kind==0?x[i]-stats[row*5]:x[i],tangent=kind==0?xt[i]-stats[row*5+1]:xt[i];
  if(mode==1){out[i]=kind==0?(value*stats[row*5+2])*w[col]+bias[col]:(value*w[col])*stats[row*5+2];}
  else{out[i]=w[col]*(tangent*stats[row*5+2]+value*stats[row*5+3]);}
 }
}

extern "C" __global__ void training_mod_dual_backward(
 const float* a,const float* at,const float* b,const float* bt,const float* gv,const float* gt,float* out,
 unsigned n,unsigned kind,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(kind==0){
  if(mode<2){if(i>=n)return;float scale=1+b[n+i];out[i]=mode==0?gv[i]*scale+gt[i]*bt[n+i]:gt[i]*scale;}
  else{if(i>=2*n)return;unsigned c=i%n;if(i<n)out[i]=mode==2?gv[c]:gt[c];else out[i]=mode==2?gv[c]*a[c]+gt[c]*at[c]:gt[c]*a[c];}
 }else{
  if(i>=n)return;
  if(mode==0)out[i]=gv[i]*b[i]+gt[i]*bt[i];else if(mode==1)out[i]=gt[i]*b[i];else if(mode==2)out[i]=gv[i]*a[i]+gt[i]*at[i];else out[i]=gt[i]*a[i];
 }
}

// nvcc -ptx -arch=sm_80 --fmad=false -ftz=false -prec-div=true -prec-sqrt=true
// --std=c++17 training_dual.cu -o training_dual.ptx
extern "C" __global__ void training_linear_dual_backward(
 const float* x,const float* xt,const float* w,const float* gv,const float* gt,float* out,
 unsigned rows,unsigned inputDim,unsigned outputDim,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;float sum=0;
 if(mode<2){
  if(i>=rows*inputDim)return;unsigned row=i/inputDim,col=i%inputDim;
  for(unsigned q=0;q<outputDim;q++)sum+=(mode==0?gv[row*outputDim+q]:gt[row*outputDim+q])*w[q*inputDim+col];
 }else if(mode==2){
  if(i>=outputDim*inputDim)return;unsigned row=i/inputDim,col=i%inputDim;
  for(unsigned q=0;q<rows;q++)sum+=gv[q*outputDim+row]*x[q*inputDim+col];
  for(unsigned q=0;q<rows;q++)sum+=gt[q*outputDim+row]*xt[q*inputDim+col];
 }else{if(i>=outputDim)return;for(unsigned q=0;q<rows;q++)sum+=gv[q*outputDim+i];}
 out[i]=sum;
}

extern "C" __global__ void training_norm_dual_backward(
 const float* x,const float* xt,const float* w,const float* gv,const float* gt,const float* stats,float* out,
 unsigned rows,unsigned width,unsigned kind,unsigned mode){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(mode>=2){
  if(i>=width)return;float sum=0;
  for(unsigned r=0;r<rows;r++){
   unsigned base=r*width;float mean=stats[r*5],dmean=stats[r*5+1],inv=stats[r*5+2],dinv=stats[r*5+3];
   if(mode==3)sum+=gv[base+i];
   else{float value=kind==0?x[base+i]-mean:x[base+i],tangent=kind==0?xt[base+i]-dmean:xt[base+i];sum+=(gv[base+i]*value)*inv+gt[base+i]*(tangent*inv+value*dinv);}
  }
  out[i]=sum;return;
 }
 if(i>=rows)return;unsigned base=i*width;
 float mean=stats[i*5],dmean=stats[i*5+1],inv=stats[i*5+2],dinv=stats[i*5+3],dvar=stats[i*5+4];
 float ginv=0,gdinv=0;
 for(unsigned c=0;c<width;c++){
  float value=kind==0?x[base+c]-mean:x[base+c],tangent=kind==0?xt[base+c]-dmean:xt[base+c];
  float g=gv[base+c]*w[c],h=gt[base+c]*w[c];ginv+=g*value+h*tangent;gdinv+=h*value;
 }
 float gvar=ginv*(-.5f)*inv*inv*inv+gdinv*.75f*inv*inv*inv*inv*inv*dvar;
 float gdvar=gdinv*(-.5f)*inv*inv*inv;float denom=kind==0?float(width):float(width-1);
 float total=0;
 if(kind==0){for(unsigned c=0;c<width;c++){
  float center=x[base+c]-mean,dcenter=xt[base+c]-dmean,g=gv[base+c]*w[c],h=gt[base+c]*w[c];
  float value=mode==0?g*inv+h*dinv:h*inv;
  if(mode==0)value+=gvar*2*center/denom+gdvar*2*dcenter/denom;else value+=gdvar*2*center/denom;
  total-=value;
 }}
 for(unsigned c=0;c<width;c++){
  float center=x[base+c]-mean,dcenter=xt[base+c]-dmean,g=gv[base+c]*w[c],h=gt[base+c]*w[c];
  float value=mode==0?g*inv+h*dinv:h*inv;
  if(mode==0)value+=gvar*2*center/denom+gdvar*2*dcenter/denom;else value+=gdvar*2*center/denom;
  if(kind==0)value+=total/float(width);out[base+c]=value;
 }
}

// Separate entry preserves legacy AdamW identity and operation order. The
// coordinator checks the four-scalar clip result before any parameter mutation.
extern "C" __global__ void training_adamw_scaled(float *p, const float *grad,
 float *moment, float *variance, float *ema, const float *clip, unsigned n,
 float lr, float b1, float b2, float eps, float decay, float bias1, float bias2,
 float emaDecay) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n) return;
 float g=__fmul_rn(grad[i],clip[3]);
 float m=__fadd_rn(__fmul_rn(b1,moment[i]),__fmul_rn(__fsub_rn(1.f,b1),g));
 float v=__fadd_rn(__fmul_rn(b2,variance[i]),__fmul_rn(__fmul_rn(__fsub_rn(1.f,b2),g),g));
 float mh=__fdiv_rn(m,bias1), vh=__fdiv_rn(v,bias2);
 float value=__fmul_rn(p[i],__fsub_rn(1.f,__fmul_rn(lr,decay)));
 value=__fsub_rn(value,__fdiv_rn(__fmul_rn(lr,mh),__fadd_rn(__fsqrt_rn(vh),eps)));
 p[i]=value; moment[i]=m; variance[i]=v;
 if(emaDecay>0.f) ema[i]=__fadd_rn(__fmul_rn(emaDecay,ema[i]),__fmul_rn(__fsub_rn(1.f,emaDecay),value));
}
