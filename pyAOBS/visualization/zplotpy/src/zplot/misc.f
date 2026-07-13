c
c     ----------------------------------------------------------------
c
      subroutine remdc(x,npts,n1,n2)
c
c     remove the dc (calculated between points n1 and n2) from the
c     array x
c
      include 'zplot.par'
      real x(npmax)
c
      xsum=0.
c
      do 10 i=n1,n2
         xsum=xsum+x(i)
10    continue
c
      if(xsum.ne.0.) then
        dc=xsum/float(n2-n1+1)
      else
        return
      end if
c
      do 20 i=1,npts
         x(i)=x(i)-dc
20    continue
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine crscor(pilot,ncrcor,x,npts,n0,nlag,lag,ratio)
c
c     determine optimal lag of array x with respect to pilot trace
c     using simple cross-correlation method
c
      include 'zplot.par'
      real pilot(npmax),x(npmax),enerp,enerx
	  real tmpp(ncrcor),tmpx(ncrcor),pweght,ratio
c
      ccmax=-1.e20
c
      enerp=0
      do 30 j=1,ncrcor
		 tmpp(j)=pilot(j)
		 enerp=enerp+tmpp(j)**2
30    continue
      call  hilbert(tmpp,ncrcor,1)

      do 10 i=-nlag,nlag,1
c
         ccsum=0.
c
         ntlag=n0+i-1
c
	     enerx=0
         do 40 j=1,ncrcor
			tmpx(j)=x(ntlag+j)
			enerx=enerx+tmpx(j)**2
40       continue
         call  hilbert(tmpx,ncrcor,1)
c

         do 20 j=1,ncrcor
            pweght=sqrt(cos(0.5*(tmpp(j)-tmpx(j)))**2)**ratio
            ccsum=ccsum+pilot(j)*x(ntlag+j)*pweght
20       continue
c
         ccsum=ccsum/sqrt(enerp*enerx)

         if(ccsum.gt.ccmax) then
           lag=i
           ccmax=ccsum
         end if
10    continue
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine mensym(x,y,albht,label,icol,ifbcol)
c
c     plot the name of a menu item
c
      character*14 label
c
      if(icol.lt.2) then
ccc        call rtxcol(ifbcol,ifbcol)
        call pcolor(ifbcol)
      else
ccc        call rtxcol(icol,icol)
        call pcolor(icol)
      end if
c
      call symbol(x,y,albht,label,0.,14)
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine mennum(x,xshift,y,albht,value,ndeci,icol,ifbcol)
c
c     plot the value of a menu item
c
      if(icol.lt.2) then
ccc        call rtxcol(ifbcol,ifbcol)
        call pcolor(ifbcol)
      else
ccc        call rtxcol(icol,icol)
        call pcolor(icol)
      end if
c
      call number(x+xshift*albht,y,albht,value,0.,ndeci)
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine sort(ra,rb,n)
c
c     sort the elements of array x in order of increasing size using
c     a heapsort technique
c
      real ra(n)
      integer rb(n)
c
      do 30 i=1,n
         rb(i)=i
30    continue
c
      l=n/2+1
      ir=n
c
10    continue
c
      if(l.gt.1) then
        l=l-1
        rra=ra(l)
        rrb=rb(l)
      else
        rra=ra(ir)
        rrb=rb(ir)
        ra(ir)=ra(1)
        rb(ir)=rb(1)
        ir=ir-1
        if(ir.eq.1) then
          ra(1)=rra
          rb(1)=rrb
          return
        end if
      end if
      i=l
      j=l+l
20    if(j.le.ir) then
        if(j.lt.ir) then
          if(ra(j).lt.ra(j+1)) j=j+1
        end if
        if(rra.lt.ra(j)) then
          ra(i)=ra(j)
          rb(i)=rb(j)
          i=j
          j=j+j
        else
          j=ir+1
        end if
        go to 20
      end if
      ra(i)=rra
      rb(i)=rrb
      go to 10
c
      end
c
c     ----------------------------------------------------------------
c
      subroutine pick(seis,npts,nstart,nfini,tmin,time,nwind,dts,
     +                iflag,minenratio)
c
c     make an arrival pick of a trace in seis by calculating the
c     ratio of the energy in a window on either side
c     of each point and selecting the point corresponding to the
c     largest ratio
c
      include 'zplot.par'
c
      real seis(npts),seis2(npmax),max,minenratio
c
      iflag=0
      nw2=nwind*2
      nit=min(npts-nw2,nfini-nw2)
      nstpnw=nstart+nwind
      if(nit-nstart.lt.2) return
      do 5 i=1,npts
         seis2(i)=seis(i)*seis(i)
5     continue
      max=minenratio
c
      sum1=0.
      sum2=0.
      do 20 j=1,nwind
         sum1=sum1+seis2(nstpnw-j)
         sum2=sum2+seis2(nstpnw+j)
20    continue
      er=sum2/sum1
      if(er.gt.max) then
        ipos=nstart
        max=er
        iflag=1
      end if
c
      do 10 i=nstart+1,nit
         im1=i-1
         sum1=sum1-seis2(im1)+seis2(im1+nwind)
         sum2=sum2-seis2(i+nwind)+seis2(i+nw2)
         er=sum2/sum1
         if(er.gt.max) then
           ipos=i
           max=er
           iflag=1
         end if
10    continue
c
      if(iflag.eq.1) time=tmin+float(ipos+nwind-1)*dts
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine vg2(x,npts,dts,tvg,pvg)
c
c     vg (variable gain) filter
c
c     apply vg to trace x by dividing each point by the sum of the
c     absolute value of the points raised to the pvg power in a
c     window of length tvg centred about that point
c
c     x ------- array containing trace to be filtered
c     npts ---- number of points in array x
c     dts --- sampling interval in seconds
c     tvg ----- filter window length in seconds
c     pvg ----- power of filter:   pvg<0  --  accentuate 'highs'
c                                  pvg=0  --  has no effect
c                                  pvg>0  --  bring up 'lows'
c
      include 'zplot.par'
      real*4 x(npts),e(npmax),xvg(npmax)
c
      nw=nint(tvg/dts)+1
      if(mod(nw,2).eq.0) nw=nw+1
      n1=nw/2+1
      if(nw.lt.1) nw=1
      if(nw.gt.npts) nw=npts
c
      do 10 i=1,npts
         e(i)=abs(x(i))**pvg
         xvg(i)=0.
10    continue
c
      esum=0.
      do 20 i=1,nw
         esum=esum+e(i)
20    continue
c
      do 30 i=1,n1
         if(esum.ne.0.) xvg(i)=x(i)/esum
30    continue
c
      if(nw.lt.npts) then
        do 40 i=n1+1,npts-n1
           esum=esum-e(i-n1)+e(i+n1-1)
           if(esum.ne.0.) xvg(i)=x(i)/esum
40      continue
        esum=esum-e(npts-nw)+e(npts)
      end if
c
      do 50 i=npts-n1+1,npts
         if(esum.ne.0.) xvg(i)=x(i)/esum
50    continue
c
      do 60 i=1,npts
         x(i)=xvg(i)
60    continue
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine template(iopentp,tpfile,xtemp,ttemp,mtemp,ntemp,
     +                    rdepthm,iflagm,iwater,vwater,sdepthm)
c
c     read in the pick template file
c
      include 'zplot.par'
      real xtemp(nptemp),ttemp(nptemp)
	  integer mtemp(nptemp),npath,ittt
      character tpfile*80
c
      iflagm=0
c
100   open(18, file=tpfile, status='old', err=99, iostat=ipfile)
99    if(ipfile.gt.0.or.tpfile.eq.'') then
        write(6,5)
5       format(/'***  there is no pick template file  ***')
        write(*,*) char(7)
        write(6,15)
15      format('enter name of pick template file: ')
        read(5,25) tpfile
25      format(a80)
        if(tpfile.eq.'') then
          iflagm=1
          return
        end if
        go to 100
      end if
c
      iopentp=1
      i=1
        select case(iwater)
  	      case(1)
  	      npath=1
		   ittt=1
  		  case(2)
  		  npath=3
		   ittt=11
  		  case(3)
  		  npath=5
		   ittt=111
  		  case default
  		   ittt=0
        end select
      depth=rdepthm-sdepthm
200   read(18,*,end=300) xtemp(i),ttemp(i),mtemp(i)
      if(iwater.gt.0) then
        h=float(npath)*(depth**2+(xtemp(i)/float(npath))**2)**.5
        twater=h/vwater
        if(mtemp(i).eq.ittt) then
		  ttemp(i)=twater
		  i=i+1
	    end if
      end if
      go to 200
300   ntemp=i-1
      if(ntemp.le.1) then
        write(*,*) char(7)
        write(6,35)
35      format(/
     +  '***  pick template must have at least 2 points  ***'/)
        iflagm=1
        return
      end if
c
      close(18)
      iopentp=0
c
      return
      end
c----------------------------------------------------------------------
c*********************************************************************
c
c     subroutine tcas
c
c     Seismology Group, Research School of Earth Sciences
c     The Australian National University
c
c     Non-linear inversion scheme for differential travel times
c
c     Uses adaptive stacking to determine optimum trace
c     alignment given some initial approximate alignment. If
c     initial alignment is model predicted (e.g. AK135), then
c     model traveltime residuals will be obtained.
c
c
c     AUTHOR:  Brian L.N. Kennett (RSES,ANU)
c              November 2002
c
c     MODIFIED: Nick Rawlinson (RSES,ANU)
c               August 2003
c
c     MODIFIED: Haibo Huang (SCSIO,CAS)
c               November 2023
c*********************************************************************
      subroutine tcas(nsi,pjgl,wb,dtcw,wl,dts,
     +                    zv,ntpk,nptsk,ipltk,tshift,ratio)
      include 'zplot.par'

c ---------------------------------------------------------------------
c     INPUT VARIABLES
c ---------------------------------------------------------------------
c     nsi = Number of stacking iterations
c     dtcw = bounds on differential time search
c     pjgl = Stack index (the Lp norm used)
c     dts = sample interval
c     stkwb,stkwl = start,length of stack window
c     nptsk = number of data points per trace
c     ipltk = number of traces
c     zv = Array containing all data in raw form
c     nst0 = initial picked samples
c     tshift = OUTPUT time shift
c ------------------------------------------------------------------
      integer nsi,nptsk,ntpk(1000),nst0
      real emax,emin,stkwb,stkwl
      real erl,zv(1000,npmax)
      real dtcw,pjgl,wb,wl
      integer nstkwb,nstkwl,ipltk,nsta
      real zssl,zscp,dts
      real zu,dtcs
c
      common /RST1/ zu(1000,npmax),zssl(npmax),zscp(npmax)
      common /RST2/ dtcs(1000),nst0(1000)
c
c     npmax = maximum data length
c     zssl = linear trace stack
c     zscp = quadratic trace stack
c     zu = Array containing all normalized data
c     dtcs = local time shift from stacking iteration
c     nstkwb = number of samples to start of stack window
c     nstkwl = number of samples in stack window
c
c ---------------------------------------------------------------------
c     OTHER VARIABLES
c ---------------------------------------------------------------------
c
      integer i,j,l,m,j1,j2,npmx
      integer jim1,jim2,imo,lu,js,jm,jmi
      integer ist,icl,swl,swr,w(1000)
      real den,err(1000),wsp(10000)
      real vu(npmax),errl,errr,tshift(1000)
      real wsl,dtmx,scu,pstakn
      real wm,ws
c
c
c     vu = Data buffer
c     ist,icl = counters for error determination
c     err = pick error estimated from trace power
c     wsp = power of weighted stack
c     swl,swr = switches for half width error determination
c     errl,errr = Half width errors (left and right)
c     w = weight applied to each trace (usually 0 or 1)
c     scu = denominator for trace normalization
c     pstakn = L2 measure of trace misfit
c     dtmx = maximum model predicted time shift
c     npmx = trace with maximum number of samples
c
c ----------------------------------------------------------------------
      emin=0.025
      emax=0.150
	  erl=1.25
	  nsta=ipltk
	  stkwb=wb
	  stkwl=wl
      DO i=1,nsta
	     dtcs(i) = 0.
		 tshift(i)=0.
	     w(i)=0
		 nst0(i)=ntpk(i)
         if(nst0(i).gt.0)w(i)=1
         scu = 0.
         err(i)=0.0
         DO j=1,nptsk
            zu(i,j) = w(i)*zv(i,j)
c			write(*,*)nst0(i),zu(i,j)
c			
            if(abs(zu(i,j)).gt.scu) scu = abs(zu(i,j))
         ENDDO
c
c        Normalise working copy zu
c
         IF(scu.gt.0.0) THEN
            DO j=1,nptsk
               zu(i,j) = zu(i,j)/scu
c			   write(*,*)nst0(i),scu,zu(i,j)
            ENDDO
         ENDIF
      ENDDO
c
      nstkwb = int(stkwb/dts)
	  nstkwl = int(stkwl/dts)
c     Stack all preliminarily aligned traces
c
      call pstack(nstkwb,nstkwl,nsta,dts,ratio)
c --------------------------------------------------------------------
c     Start the adaptive stacking procedure
c --------------------------------------------------------------------
      jim1 = -1 * nint(dtcw/dts)
      jim2 = nint(dtcw/dts)
c	  write(*,*)'jim',jim1,jim2
      DO m = 1,nsi
         DO i = 1,nsta
		 
            IF(w(i).eq.0) THEN
               dtcs(i) = 0.
               GOTO 19
            ENDIF
			imo = nst0(i)
            wm = 1.e6
            DO js = jim1,jim2
               ws = 0.
               DO l = 1,nstkwl
                  lu = l-nstkwb+imo+js-1
c				  write(*,*)js,zssl(l),zu(i,lu)
                  ws = ws + abs(zssl(l)-zu(i,lu))**pjgl
               ENDDO
               ws = ws/stkwl
               wsp(js+1-jim1)=ws
               IF(ws.lt.wm) THEN
                  wm = ws
                  jm = js
               ENDIF
            ENDDO
c			write(*,*)'final wm,jm',wm,jm
c
c           Below, we estimate error from the power
c           of the stacked trace.
c
            if(m.eq.nsi)then
c
c              Determine the width of the stack
c              and tag the location of the minimum
c
               ist=jim2-jim1+1
               jmi=jm+1-jim1
c
c              Calculate the cross-over point for <jmi
c
               if(jmi.eq.1)then
                  swl=2
               else
                  l=jmi-1
                  swl=0
                  do while(swl.eq.0)
                     if(wsp(l).ge.wsp(jmi)*erl)then
                        icl=l
                        swl=1
                     else
                        l=l-1
                        if(l.eq.0)swl=2
                     endif
                  enddo
                  if(swl.eq.1)then
                     den=wsp(icl+1)-wsp(icl)
                     if(abs(den).gt.1.0e-5)then
                        errl=dts*(wsp(jmi)*erl-wsp(icl))/den
                        errl=(jmi-icl)*dts-errl
                     else
                        errl=(jmi-icl)*dts
                     endif
                  endif
               endif
c
c              Calculate cross-over point for >jmi
c
               if(jmi.eq.ist)then
                  swr=2
               else
                  l=jmi+1
                  swr=0
                  do while(swr.eq.0)
                     if(wsp(l).ge.wsp(jmi)*erl)then
                        icl=l
                        swr=1
                     else
                        l=l+1
                        if(l.eq.ist)swr=2
                     endif
                  enddo
                  if(swr.eq.1)then
                     den=wsp(icl)-wsp(icl-1)
                     if(abs(den).gt.1.0e-5)then
                        errr=dts*(wsp(jmi)*erl-wsp(icl-1))/den
                        errr=(icl-jmi-1)*dts+errr
                     else
                        errr=(icl-jmi)*dts
                     endif
                  endif
               endif
c
c              Take average of errr and errl for actual error
c
               if(swr.eq.1.and.swl.eq.1)then
                  err(i)=(errr+errl)/2.0
               else if(swl.eq.1)then
                  err(i)=errl
               else if(swr.eq.1)then
                  err(i)=errr
               else
                  err(i)=emax
               endif
c
c              Constrain error limits
c
               if(err(i).lt.emin)then
                  err(i)=emin
               else if(err(i).gt.emax)then
                  err(i)=emax
               endif
            endif
c
c           Determine the time shift
c
            dtcs(i) = float(jm)*dts
			tshift(i) = dtcs(i)
c			write(6,*)'tshift=',tshift(i)
 19         CONTINUE
         ENDDO
         call pstack(nstkwb,nstkwl,nsta,dts,ratio)

      ENDDO
	  return
      end 
*********************************************************************
      subroutine pstack(pnkwb,pnkwl,pnsta,pdts,ratio)
      include 'zplot.par'
*********************************************************************
c
c     Perform a linear and quadratic stack of all traces
c
c--------------------------------------------------------------------
c     COMMON BLOCK DECLARATIONS
c--------------------------------------------------------------------
c
      integer pnkwb,pnkwl,pnsta,nst0
      real zssl,zscp,pdts
      real zu,dtcs
	  real tmpp(npmax)
c
      common /RST1/ zu(1000,npmax),zssl(npmax),zscp(npmax)
      common /RST2/ dtcs(1000),nst0(1000)
c
c     npmax = maximum datalength
c     zssl = linear trace stack
c     zscp = quadratic trace stack
c     dts = sample interval
c     zu = Array containing all normalized data
c     dtcs = local time shift from stacking iteration
c     nst0 = initial picked samples
c     pnkwb = number of samples to start of stack window
c     pnkwl = number of samples in stack window
c     nsta = number of stations
c
c---------------------------------------------------------------------
c     OTHER VARIABLES
c---------------------------------------------------------------------
c
      integer i,l,lmn
      real zcu(npmax),scz,pstakn,tempf(npmax)
	  real tempc(npmax),temps(npmax)
c     pstakn = L2 measure of trace misfit
c---------------------------------------------------------------------
      DO l=1,pnkwl
         zssl(l) = 0.
         zscp(l) = 0.
		 tempc(l)= 0.
         temps(l)= 0.
		 tempf(l)= 0.
      ENDDO
c     loop on stations
c

		nst=0
      DO i = 1,pnsta
	    IF(nst0(i).gt.0)then
		 nst=nst+1
         lmn = -pnkwb + nst0(i) + nint(dtcs(i)/pdts) - 1

c        linear and quadratic stack
c
         DO l=1,pnkwl
		     zcu(l) = zu(i,lmn+l)
			tmpp(l) = zu(i,lmn+l)
            zssl(l) = zssl(l) + zcu(l)
            zscp(l) = zscp(l) + zcu(l)*zcu(l)
         ENDDO
c----------------------------------------		 
		 call  hilbert(tmpp,pnkwl,1)
		 DO l=1,pnkwl
            tempc(l) = tempc(l)+ cos(tmpp(l))
	    	temps(l) = temps(l)+ sin(tmpp(l))
         ENDDO
c-----------------------------------------		 
		ENDIF
      ENDDO
	  DO l=1,pnkwl
	  tempf(l)=sqrt(tempc(l)**2+temps(l)**2)/nst
	  tempf(l)=tempf(l)**ratio
      ENDDO

c      write(*,*)nst,zscp(pnkwl)
      pstakn = 0.
      DO l=1,pnkwl
         pstakn = pstakn + abs(zscp(l))
      ENDDO
      pstakn = pstakn/(real(nst)*pnkwl)
	  write(6,*) "pstakn = ", pstakn
      DO l=1,pnkwl
         zssl(l) = zssl(l)*tempf(l)/(real(nst))
      ENDDO	  
      scz = 0.
      DO l=1,pnkwl
         IF(abs(zssl(l)).gt.scz) scz = abs(zssl(l))
      ENDDO
      IF(scz.gt.0.0) THEN
         DO l=1,pnkwl
            zssl(l) = zssl(l)/scz
         ENDDO
      ENDIF

      return
      end

c
c     ----------------------------------------------------------------	  
      subroutine smooth(x,n) 
c                 
c     three point triangular smoothing filter
c                 
      real x(n) 
      m=n-1       
      a=0.77*x(1)+0.23*x(2) 
      b=0.77*x(n)+0.23*x(m) 
      xx=x(1)     
      xr=x(2)     
      do 10 i=2,m 
         xl=xx    
         xx=xr    
         xr=x(i+1) 
         x(i)=0.54*xx+0.23*(xl+xr) 
 10   continue    
      x(1)=a      
      x(n)=b      
      return      
      end      	  