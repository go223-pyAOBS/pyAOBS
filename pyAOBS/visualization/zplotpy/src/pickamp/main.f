c
c
c     A program to estimate pick uncertainties
c
      include 'zplot.par'
      parameter (npair=6, nlines=nsmax)
c
      integer fhead(13),sint,tstart,tend,thead(22+npickw),
     +        pick(npickw),selev,swdepth,relev,sxutm,syutm,
     +        sz,rxutm,ryutm,rz,it1(nlines),it2(nlines),
     +        numamp(npickw),pnum1(npair),pnum2(npair),
     +        type(npickw)/npickw*1/
      real frhead(13),seisdc(npmax),seis(npmax),picks(npickw)
	  real maxamp,minamp,ampave(npickw)
      character dfile*80,fmt*10,hfile*80,ofile*80,ohfile*80
c
      equivalence (ntraces ,fhead(1)),(npts, fhead(2)),
     +            (sint, fhead(3)),(tstart,fhead(4)),
     +            (tend,fhead(5)),(nrec,fhead(6)),
     +            (npick,fhead(7)),(vredf,frhead(8)),
     +            (ifmt,fhead(9)),(xlatlong,frhead(10)),
     +            (xelev,frhead(11)),(xutm,frhead(12)),
     +            (cm,frhead(13)),(fhead,frhead)
c
      common/hedu/fhead,thead,nbytes,picks
c
      namelist /picpar/ dfile,hfile,ofile,ohfile,twin,pick,imeth
	  namelist /fltpar/ ibndps,freqlo,freqhi,izerop,npoles
c
      data pick,ampave,numamp/npickw*-1,npickw*0.,npickw*0/
c
      twin=.25
	  imeth=0
      nc=0
      npro=0
      nrecc=0
	  ibddd=0
      ibndps=0
      freqlo=3.
      freqhi=15.
      izerop=1
      npoles=8	
	  
      ofile='pickamp.out'
	  ohfile='pickamp.hdr'
c
c
      open(unit=15, file='pickamp.in', status='old')
c
      read(15,picpar)
	  read(15,fltpar)
c
      do 410 i=1,npickw
         if(pick(i).lt.1.or.pick(i).gt.npickw) go to 430
         npicko=npicko+1
410   continue
      npicko=npickw
430   continue
c
c
      open(10,file=dfile,status='old',access='direct',recl=52)
c
      read(10,rec=1) fhead
c
      open(12, file=hfile, status='old',form='unformatted')
      open(14, file=ohfile,form='unformatted')
      open(unit=11, file=ofile)
c
      do 310 i=1,ntraces
         read(12) nrec,itsn,irec,itype,iflag,
     +      offst,azi,igword,t1,slat,slong,
     +      selev,swdepth,rlat,rlong,relev,sxutm,syutm,sz,
     +      rxutm,ryutm,rz,(picks(j),j=1,npick)
c	    write(6,72)(picks(j),j=1,npick)
c 72      format(20f10.5/)	
         nc=nc+1 
         if(nrec.ne.nrecc) then
           npro=npro+1
           nrecc=nrec
           it1(npro)=nc
           if(npro.gt.1) it2(npro-1)=nc-1
         end if
310   continue
      rewind(12)
c
      it2(npro)=nc
c 
      write(6,95) nc
95    format(/'number of lines in header file: ',i10/)
      write(6,85) npro
85    format('number of profiles: ',i5)
      write(6,75) (it1(i),it2(i),i=1,npro)
75    format('first trace    last trace'/500(2i10)/)
c
      if(vredf.eq.0.) then
         rvredf=0.
      else
         rvredf=1./vredf
      end if
      sints=float(sint)/1000000.
      if(ifmt.eq.1) then
        nbytes=npts*4+(22+npick)*4
        fmt='    real*4'
c       stop
      else
        nbytes=npts*2+(22+npick)*4
        fmt=' integer*2'
      end if
c
      write(6,105) dfile,ntraces,npts,float(sint)/1000000.,
     +             float(tstart)/1000.,float(tend)/1000.,
     +             nrec,npick,vredf,fmt
105   format(/'data file opened: '/,
     +'     file name:               ',a80/
     +'     number of traces:        ',i10/
     +'     points per traces:       ',i10/
     +'     sampling interval (s):   ',f10.5/
     +'     start/end times (s):     ',2f10.5/
     +'     number of records:       ',i10/
     +'     number of pick words:    ',i10/
     +'     reducing velocity (km/s):',f10.3/
     +'     data format:             ',a10)
c
      tminf=float(tstart)/1000.
      tmaxf=float(tend)/1000.
      nwin=nint(twin/sints)
      if(npts.lt.2) then
        write(*,60)
60      format(/'***  must be at least 2 points per trace  **'/)		 
        write(*,*) char(7)
        stop
      end if

c       
      close(10)
      open(10,file=dfile,access='direct',recl=nbytes)
c
	
      write(6,33)
      write(11,33)
33    format(/'    offset    amplitude  pick')
c
      do 10 ipro=1,npro
      ampsum=0.
      namp=0  
      do 110 itn=it1(ipro),it2(ipro)
c
         iflagb=0
         read(12) nrec,itsn,irec,itype,iflag,
     +      offst,azi,igword,t1,slat,slong,
     +      selev,swdepth,rlat,rlong,relev,sxutm,syutm,sz,
     +      rxutm,ryutm,rz,(picks(j),j=1,npick)
         offst=offst/1000.
c
         call gettrc(itn,seis)
c
         call remdc(seis,npts,1,npts)
         if(ibndps.eq.1) then
c 		  write(6,*)freqlo,freqhi,npoles  
		  call bndpas(freqlo,freqhi,sints,npoles,izerop,seis,npts,iflagb)		 
         end if
c
         do 340 i=1,npicko
            if(picks(pick(i)).gt.0.) then
              tpick=picks(pick(i))
              n0=nint((tpick-abs(offst)*rvredf-tminf)/sints)+1
			  n1=n0+nwin
			  if(n0.lt.1) n0=1
              if(n1.lt.1) n1=1
              if(n0.gt.npts) n0=npts
              if(n1.gt.npts) n1=npts
			  nwina=n1-n0+1
              maxamp=0.
			  minamp=0.
              if(n1.gt.n0) then
                if(imeth.eq.2) then
                  do 360 jj=n0,n1
                     if(seis(jj).lt.minamp) minamp=seis(jj)
                     if(seis(jj).gt.maxamp) maxamp=seis(jj)
360               continue
                  maxamp=(abs(minamp)+abs(maxamp))/2.
                else if(imeth.eq.1) then
                  sum=0.
                  do 361 jj=n0,n1
                     sum=sum+abs(seis(jj))
361               continue
                  maxamp=sum/float(nwina)
                else
				  sum=0.
			      do 362 jj=n0,n1
					 sum=sum+seis(jj)**2
362				  continue
                  maxamp=sqrt(sum/float(nwina))
				end if
               end if
			   
			  write(6,77),offst,maxamp,pick(i) 
			  write(11,77),offst,maxamp,pick(i) 
77			  format(f10.3, f12.3, i5)

			  picks(pick(i))=maxamp
              numamp(i)=numamp(i)+1
              ampave(i)=ampave(i)+maxamp
              namp=namp+1
              ampsum=ampsum+maxamp
			
			else
              picks(pick(i))=0.
            end if 
340      continue
c

c
         do 350 i=1,npick
            do 370 j=1,npicko
               if(i.eq.pick(j)) go to 350
370         continue
            picks(i)=0.
350      continue
c
c    haibo
         offst=offst*1000.
         write(14) nrec,itsn,irec,itype,iflag,
     +      offst,azi,igword,t1,slat,slong,
     +      selev,swdepth,rlat,rlong,relev,sxutm,syutm,sz,
     +      rxutm,ryutm,rz,(picks(j),j=1,npick)
c
110   continue
      write(6,1)
      write(11,1)
1     format(/'record    ntrace     npick   ave amp')
      if(namp.gt.0) then
      write(6,2) ipro,it2(ipro)-it1(ipro)+1,namp,ampsum/namp
      write(11,2) ipro,it2(ipro)-it1(ipro)+1,namp,ampsum/namp
2     format(i6,2i10,f12.3)
      else
      write(6,2) ipro,it2(ipro)-it1(ipro)+1,namp,0.
      write(11,2) ipro,it2(ipro)-it1(ipro)+1,namp,0.
      end if
10    continue
c
      close(10)
c
      write(6,45)
	  write(11,45)
45	  format(/'  pick    number   ave_amp')
c
      do 510 i=1,npicko
         if(numamp(i).gt.0) then
		 ampave(i)=ampave(i)/float(numamp(i))
		 end if
510   continue
c
      write(6,55)(pick(i),numamp(i),ampave(i),i=1,npicko)
      write(11,55)(pick(i),numamp(i),ampave(i),i=1,npicko)
55	  format(i6,i10,f12.4)

c
c
      if(namp.gt.0) then
        write(6,3) namp,ampsum/namp,ntraces
        write(11,3) namp,ampsum/namp,ntraces
3       format(/' total number of picks: ',i8/ 
     +         '         ave amplitude: ',f12.2/
     +         'total number of traces: ',i8)
      end if
c 

      write(6,*) '####################################'
	  write(11,*) '####################################'
      if(imeth.eq.2) then
	    write(6,*)  ' amplitude = (|max| + |min|) / 2'
	    write(11,*) ' amplitude = (|max| + |min|) / 2'
      else if(imeth.eq.1) then
	    write(6,*)  ' amplitude = sum(|amp|)/ N'
	    write(11,*) ' amplitude = sum(|amp|)/ N'
      else	   
	    write(6,*)  ' amplitude = sqrt(sum(amp**2)) / N'
	    write(11,*) ' amplitude = sqrt(sum(amp**2)) / N'
	  end if
	  write(6,*)  '####################################'	
	  write(11,*) '####################################'	    
      stop
      end
c
c     -------------------------------------------------------------------
c

