c                 
c     ---------------------------------------------------------------
c                 
      subroutine bndpas(freqlc,freqhc,deltat,npole,izp,x,n,iflag)
c
c     written by D. White years ago
c                 
c     subroutine bndpas is a modified version of a subroutine 
c     taken from 'time sequence analysis in geophysics', 3rd ed. by 
c     e.r. kanasewich. the purpose of this subroutine is to design 
c     and apply a recursive butterworth band pass filter. in 
c     order to design the filter a call must be made to bnd- 
c     pas and then the filter may be applied by calls to fil- 
c     ter. the filter will have n poles in the s-plane and can
c     be applied in the forward direction (non-zero phase) or 
c     in the forward and reverse directions so as to 
c     have zero phase. the cutoff frequencies will be 6db down
c     and the rolloff will be about 96 db per octave. a bi- 
c     linear z-transform is used in designing the filter to 
c     prevent aliasing problems. 
c                 
c     freqlc, freqhc -- low and high cut frequencies in hz
c     deltat ---------- sampling interval in seconds
c     npole ----------- number of poles of butterworth filter 
c     izp ------------- 1 implements the filter as a zero phase filter
c     x --------------- data vector of length n containing data
c                       to be filtered
c     iflag ----------- 0 means first time routine called, otherwise
c                       equals 1
c                 
      implicit real*8(a-h,o-z)
      complex*16 p(64),s(128),z1,z2 
      real*8 d(128) 
      real*4 x(n),freqlc,freqhc,deltat
      common /blkbp/ d,g
      data pi/3.14159265358979312/
c                 
      ig=1        
      if(iflag.eq.1) go to 100
      iflag=1     
      f1=freqlc   
      f2=freqhc   
      delt=1000.d0*deltat
c                 
      twopi=2.0d0*pi
      dt=delt/1000.0d0
      tdt=2.0d0/dt 
      fdt=4.0d0/dt 
      npl=npole/2 
      do 10 i=1,npl 
         ind=(1+(-1)**i)/2*(npl-i/2+1)+(1+(-1)**(i+1))/2*(i/2+1) 
         arg=twopi*(2*ind-1)/(4.*npl) 
         pr=-dsin(arg) 
         pii=dcos(arg) 
         p(i)=dcmplx(pr,pii) 
10    continue    
      w1=twopi*f1 
      w2=twopi*f2 
      w1=tdt*dtan(w1/tdt) 
      w2=tdt*dtan(w2/tdt) 
      hwid=(w2-w1)/2.0d0
      ww=w1*w2    
      do 19 i=1,npl 
         z1=p(i)*hwid 
         z2=z1*z1-ww 
         z2=cdsqrt(z2) 
         s(i)=z1+z2 
         s(i+npl)=z1-z2 
19    continue    
      g=.5d0/hwid 
      g=g**npl    
      npole1=npole-1 
      do 29 i=1,npole1,2 
         b=-2.0d0*dreal(s(i)) 
         z1=s(i)*s(i+1) 
         c=dreal(z1) 
         a=tdt+b+c/tdt 
         g=g*a    
         d(i)=(c*dt-fdt)/a 
         d(i+1)=(a-2.0d0*b)/a 
29    continue    
      g=g*g       
c                 
100   call filter(x,n,ig,izp,npole)
c                 
      return      
      end         
c                 
c     ----------------------------------------------------------------
c                 
      subroutine filter(x,n,ig,izp,npole) 
c                 
c     subroutine filter applies the butterworth bndpass filter 
c     designed by subroutine bndpas to the time series x.  the 
c     filter can be applied in the forward and reverse direc- 
c     so as to have zero phase, or it may be applied in the 
c     forward direction only. 
c                 
c     x ------ data vector of length n containing data to be filtered 
c     ig ----- 1 means to remove the filter gain so that the gain is 
c              unity 
c     izp ---- 1 implements the filter as a zero phase filter 
c     npole -- the number of poles of the butterworth filter 
c              designed by subroutine bndpas 
c                 
      dimension x(n),y(128,3)
      real*8 d(128),g
      common /blkbp/ d,g
c                 
c     apply filter in forward direction 
c                 
      continue    
      npole1=npole-1 
      npl=npole/2 
      nplm1=npl-1 
      xm2=x(1)    
      xm1=x(2)    
      xm=x(3)     
      y(1,1)=xm2  
      y(1,2)=xm1-d(1)*y(1,1) 
      y(1,3)=xm-xm2-d(1)*y(1,2)-d(2)*y(1,1) 
      do 10 i=2,nplm1 
         y(i,1)=y(i-1,1) 
         y(i,2)=y(i-1,2)-d(2*i-1)*y(i,1) 
         y(i,3)=y(i-1,3)-y(i-1,1)-d(2*i-1)*y(i,2)-d(2*i)*y(i,1) 
10    continue    
      x(1)=y(nplm1,1) 
      x(2)=y(nplm1,2)-d(npole1)*x(1) 
      x(3)=y(nplm1,3)-y(nplm1,1)-d(npole1)*x(2)-d(npole)*x(1) 
      do 39 i=4,n 
         xm2=xm1  
         xm1=xm   
         xm=x(i)  
         k=i-((i-1)/3)*3 
         go to (34,35,36),k 
34       m=1      
         m1=3     
         m2=2     
         go to 37 
35       m=2      
         m1=1     
         m2=3     
         go to 37 
36       m=3      
         m1=2     
         m2=1     
37       y(1,m)=xm-xm2-d(1)*y(1,m1)-d(2)*y(1,m2) 
         do 12 l=2,nplm1 
            y(l,m)=y(l-1,m)-y(l-1,m2)-d(2*l-1)*y(l,m1)-d(2*l)*y(l,m2) 
12       continue 
         x(i)=y(nplm1,m)-y(nplm1,m2) 
         x(i)=x(i)-d(npole1)*x(i-1)-d(npole)*x(i-2) 
39    continue    
      if(izp.ne.1) go to 499
c                 
c     filter in reverse direction
c                 
      xm2=x(n)    
      xm1=x(n-1)  
      xm=x(n-2)   
      y(1,1)=xm2  
      y(1,2)=xm1-d(1)*y(1,1) 
      y(1,3)=xm-xm2-d(1)*y(1,2)-d(2)*y(1,1) 
      do 14 i=2,nplm1 
         y(i,1)=y(i-1,1) 
         y(i,2)=y(i-1,2)-d(2*i-1)*y(i,1) 
         y(i,3)=y(i-1,3)-y(i-1,1)-d(2*i-1)*y(i,2)-d(2*i)*y(i,1) 
14    continue    
      x(n)=y(nplm1,1) 
      x(n-1)=y(nplm1,2)-d(npole1)*x(n) 
      x(n-2)=y(nplm1,3)-y(nplm1,1)-d(npole1)*x(n-1)-d(npole)*x(n) 
      do 49 i=4,n 
         xm2=xm1  
         xm1=xm   
         j=n-i+1  
         xm=x(j)  
         k=i-((i-1)/3)*3 
         go to (44,45,46),k 
44       m=1      
         m1=3     
         m2=2     
         go to 47 
45       m=2      
         m1=1     
         m2=3     
         go to 47 
46       m=3      
         m1=2     
         m2=1     
47       y(1,m)=xm-xm2-d(1)*y(1,m1)-d(2)*y(1,m2) 
         do 16 l=2,nplm1 
            y(l,m)=y(l-1,m)-y(l-1,m2)-d(2*l-1)*y(l,m1)-d(2*l)*y(l,m2) 
16       continue 
         x(j)=y(nplm1,m)-y(nplm1,m2) 
         x(j)=x(j)-d(npole1)*x(j+1)-d(npole)*x(j+2) 
49    continue    
499   if(ig.ne.1) return 
      gfac=g      
      if(izp.eq.0) gfac=sqrt(gfac)
      do 59 i=1,n 
         x(i)=x(i)/gfac
59    continue    
      return      
      end         
