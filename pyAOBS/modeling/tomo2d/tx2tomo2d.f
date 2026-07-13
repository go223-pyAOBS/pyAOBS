c     File patched for g77 compilation
c     Scott Pearse / email: scott.pearse@gmail.com / web: http://www.linkedin.com/in/scottpearse
c
c
c     version 1.2  Mar 1992
c
c     ----------------------------------------------------------------
c     |                                                              |
c     |               ********  TX2TOMO2D  ********                  |   
c     |                                                              |
c     |         Select particular phases from a "tx.in" file         |   
c     |                                                              |
c     |                   Written by C. A. Zelt                      |
c     |                    Modified by Haibo Huang                   |
c     |                Geological Survey of Canada                   |   
c     |                  Ottawa, Canada K1A 0Y3                      |
c     |                                                              |
c     ----------------------------------------------------------------
c
c
      parameter(nphase=50,ntmax=10000,nsmax=100)
      integer phase1(nphase),phase2(nphase),ishot(nsmax),
     +        npick(nsmax),ntrf(nsmax),ntrl(nsmax)
	  real xshot(nsmax),zshot(nsmax),
     +     rx1(nsmax,ntmax),rt1(nsmax,ntmax),ru1(nsmax,ntmax),
     +     rx2(nsmax,ntmax),rt2(nsmax,ntmax),ru2(nsmax,ntmax)	 
      data phase1/nphase*0/,phase2/nphase*0/,
     +     npick/nsmax*0/,ntrf/nsmax*0/,ntrl/nsmax*0/
      character txin*80,stin*80,txout1*80,txout2*80

  
      nshot=0
      nprf=1
	  nprl=1
c

995   format(a80)
904   format('Enter input station file: ')
905   format('Enter input   tx.in file: ')
906   format('Enter output   time file: ')
907   format('Enter output   geom file: ')
      write(6,904) 
      read(5,995) stin
      write(6,905) 
      read(5,995) txin
      write(6,906) 
      read(5,995) txout1
      write(6,907) 
      read(5,995) txout2	  
	  if(stin.eq.'')stin='station.lis'
	  if(txin.eq.'')txin='tx.in'
	  if(txout1.eq.'')txout1='ttimes.dat'
	  if(txout2.eq.'')txout2='geom.data'
c     write(*,*)stin,txin,txout1,txout2	  
      open(10, file=stin,status='old')
      open(11, file=txin,status='old')
      open(12, file=txout1)
      open(13, file=txout2)	  
c
      write(6,15) 
15    format(/'Enter refraction phase number (0 to stop)')
1001  read(5,*) phase1(nprf)
      if(phase1(nprf).gt.0) then
        nprf=nprf+1
        go to 1001
      end if
      nprf=nprf-1
c
      write(6,16) 
16    format(/'Enter reflection phase number (0 to stop)')
1002  read(5,*) phase2(nprl)
      if(phase2(nprl).gt.0) then
        nprl=nprl+1
        go to 1002
      end if
      nprl=nprl-1
c	  
      if(nprf.eq.0) then
        write(6,25)
25      format(/'***  no refraction phases selected  ***'/)
c       stop
      end if
      if(nprl.eq.0) then
        write(6,26)
26      format(/'***  no reflection phases selected  ***'/)
c       stop
      end if	  
c
101   write(*,*)'number of refraction and reflection phases:',nprf,nprl
c
      ni=1
80    xshot(ni)=0.
      zshot(ni)=0.
      read(10,*,end=99) ishot(ni),xshot(ni),zshot(ni)
      ni=ni+1	  
      go to 80
99    nshoti=ni-1
c
      write(*,27) nshoti
27    format(/'Number of input station:',i3)	 
c 
c      do 333 i=1,nshoti
c		  write(6,33)ishot(i),xshot(i),zshot(i)
c 333   continue		  
33    format('test',i3,2f10.3)  

c
100   read(11,*,end=999) x,t,u,i
c
      if(i.lt.0) go to 991
c	
      if(i.eq.0) then
	   do 30 j=1,nshoti
	    if(abs(x-xshot(j)).le.0.001)then
		isw=ishot(j)
c		write(6,33)isw,xshot(j),zshot(j)
		end if
30     continue		
       go to 100
      end if
c
      if(i.gt.0) then
        iflagrf=0
        iflagrl=0		
        do 10 j=1,nprf
           if(i.eq.phase1(j)) iflagrf=1
10      continue
        do 20 j=1,nprl
           if(i.eq.phase2(j)) iflagrl=1
20      continue

        if(iflagrf.eq.1.or.iflagrl.eq.1) then
          if(iflagrf.eq.1) then
		  ntrf(isw)=ntrf(isw)+1
		  k1=ntrf(isw)
		  rx1(isw,k1)=x
		  rt1(isw,k1)=t
		  ru1(isw,k1)=u
		  end if
          if(iflagrl.eq.1) then
		  ntrl(isw)=ntrl(isw)+1
		  k2=ntrl(isw)
		  rx2(isw,k2)=x
		  rt2(isw,k2)=t
		  ru2(isw,k2)=u 
		  end if
          npick(isw)=npick(isw)+1
        end if
        go to 100
      end if
c
 
991   nshot=0
	  ntime=0
      do 39 i=1,nshoti
	    isw=ishot(i)
	    if(npick(isw).gt.0)then
	     nshot=nshot+1
		 ntime=ntime+npick(isw)
        end if
39    continue	
	  write(12,*) nshot
	  write(13,*) nshot	  
      do 40 i=1,nshoti
	    isw=ishot(i)
	    if(npick(isw).gt.0)then
   	     write(12,7) xshot(i),zshot(i)+0.01,npick(isw)
   	     write(13,7) xshot(i),zshot(i)+0.01,npick(isw)		 
	     do 50 j=1,ntrf(isw)
	       write(12,5) rx1(isw,j),0.01,0,rt1(isw,j),ru1(isw,j)
		   write(13,5) rx1(isw,j),0.01,0,0.,0.
50       continue
	     do 60 j=1,ntrl(isw)
	       write(12,5) rx2(isw,j),0.01,1,rt2(isw,j),ru2(isw,j)
		   write(13,5) rx2(isw,j),0.01,1,0.,0.
60       continue
        end if
40    continue	   
c

5     format('r',2f10.3,i5,2f10.3)
7     format('s',2f10.3,i5)
c
999   write(*,*)'number of picks: ',ntime
      write(*,*)'number of shots: ',nshot
c
      stop
      end
