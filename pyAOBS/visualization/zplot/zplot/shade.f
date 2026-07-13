c                 
c     ----------------------------------------------------------------
c                 
      subroutine intrtr(x,n1,n2)
c                 
c     through linear interpolation re-sample the array x consisting
c     of n1 points so that it consists of n2 points
c                 
      include 'zplot.par'
      real x(npmax),y(npmax),inc,m
c
      scale=float(n1-1)/float(n2-1)
c                 
      y(1)=x(1)   
      y(n2)=x(n1) 
c                 
      do 10 i=2,n2-1
         sf=scale*float(i-1) 
         intsf=int(sf)
         inc=sf-float(intsf)
         ip=intsf+1
         m=x(ip+1)-x(ip) 
         y(i)=m*inc+x(ip)
10    continue    
c                 
      do 20 i=1,n2
         x(i)=y(i)
20    continue    
c                 
      return      
      end         
c                 
c     ----------------------------------------------------------------
c                 
      subroutine shade(sp,tp,np,ishade,ox,xscale,orig,tinch,dens,
     +                 shadedc) 
c                 
c     shade the positive (ishade>0) or negative (ishade<0) peaks of the
c     trace using horizontal lines
c                 
c     sp      --  amplitudes in plot units 
c     tp      --  time coordinates of amplitudes 
c     np      --  number of points
c                 
      include 'zplot.par'
      real*4 sp(npmax),tp(npmax),sshade(2*npmax),tshade(2*npmax)
c
      fill=float(sign(1,ishade))
      dc=ox/xscale+orig
      dcf=dc*fill+shadedc/xscale
c
      if(abs(ishade).eq.1) call line(sp,tp,np)
c
      nshade=min(npmax,nint(dens*tinch)+1)
c
      call intrtr(tp,np,nshade)
      call intrtr(sp,np,nshade)
c
      ipeak=0
      npeak=0
      do 20 i=1,nshade
         if(fill*sp(i).gt.dcf) then
           npeak=npeak+1
           sshade(npeak)=dc
           tshade(npeak)=tp(i)
           npeak=npeak+1
           sshade(npeak)=sp(i)
           tshade(npeak)=tp(i)
           ipeak=1
         else
           if(ipeak.eq.1) then
             call line(sshade,tshade,npeak)      
             npeak=0
             ipeak=0
           end if
         end if
20    continue    
c
      if(ipeak.eq.1) call line(sshade,tshade,npeak)      
c                 
      return      
      end         
