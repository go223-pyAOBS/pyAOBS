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
     +        numunc(npickw),pnum1(npair),pnum2(npair),
     +        type(npickw)/npickw*1/
      real frhead(13),snr1(npair),picku1(npair),
     +     snr2(npair),picku2(npair),seisdc(npmax),
     +     seis(npmax),picks(npickw),uncave(npickw)
	  real snrave(npickw)
      character dfile*80,fmt*10,hfile*80,ofile*80
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
      namelist /picpar/ dfile,hfile,ofile,twin,type,
     +                  pick,snr1,picku1,tlag,snr2,picku2
	  namelist /fltpar/ ibndps,freqlo,freqhi,izerop,npoles
c
      data pick,uncave,numunc/npickw*-1,npickw*0.,npickw*0/
	  data snrave/npickw*0./
c
      tlag=.1
      twin=.25
      nc=0
      npro=0
      nrecc=0
	  ibddd=0
      ibndps=0
      freqlo=3.
      freqhi=15.
      izerop=1
      npoles=8	
	  
      ofile='pickunc.out'
c
      do 670 i=1,npair
         snr1(i)=-1.
         picku1(i)=-1.
         pnum1(i)=0
         snr2(i)=-1.
         picku2(i)=-1.
         pnum2(i)=0
670   continue
c
      open(unit=15, file='pickunc.in', status='old')
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
      np1=0
      do 380 i=1,npair
         if(snr1(i).lt.0.) then
           np1=i-1
           go to 390
         end if
380   continue
390   if(np1.eq.0) then
        write(*,40)
40      format(/'***  no snr1/picku1 pairs specified  ***'/)	 
        stop
      end if
      np2=0
      do 480 i=1,npair
         if(snr2(i).lt.0.) then
           np2=i-1
           go to 490
         end if
480   continue
490   if(np2.eq.0) then
        write(*,50)
50      format(/'***  no snr2/picku2 pairs specified  ***'/)	
        stop
      end if
c
      open(10,file=dfile,status='old',access='direct',recl=52)
c
      read(10,rec=1) fhead
c
      open(12, file=hfile, status='old',form='unformatted')
      open(14, file='pickunc.hdr',form='unformatted')
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
      nlag=nint(tlag/sints)
c      npts=nint((tmaxf-tminf)/sints)+1
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
      write(6,1)
      write(11,1)
1     format(/'record    ntrace     npick   ave snr')
c
      do 10 ipro=1,npro
      snrsum=0.
      nsnr=0
      snrsumt=0.
      nsnrt=0	  
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
c        do 440 i=1,npts
c40         seis(i)=seis(i)**2 
c
         do 340 i=1,npicko
            if(picks(pick(i)).gt.0.) then
              tpick=picks(pick(i))
              n00=nint((tpick-abs(offst)*rvredf-tminf)/sints)+1
              snrmax=0.
              do 450 k=-nlag,nlag
                 n0=n00+k
                 n1=n0-nwin
                 n2=n0+nwin
                 if(n1.lt.2) n1=2
                 if(n2.gt.npts) n2=npts
                 enern=0.
                 eners=0.
c
                 sum=0.
                 do j=n1-1,n0-1
                    sum=sum+seis(j)
                 enddo
                 ave=sum/float(n0-n1+1)
                 do j=n1-1,n0-1
                    seisdc(j)=seis(j)-ave
                 enddo
                 do 650 j=n1-1,n0-1
                    enern=enern+seisdc(j)**2
650              continue
c
                 sum=0.
                 do j=n0,n2
                    sum=sum+seis(j)
                 enddo
                 ave=sum/float(n2-n0+1)
                 do j=n0,n2
                    seisdc(j)=seis(j)-ave
                 enddo
                 do 360 j=n0,n2
                    eners=eners+seisdc(j)**2
360              continue
c
                 if(enern.gt.0) then
                   snrt=eners/enern
                 else
                   snrt=0.
                 end if
                 if(snrt.gt.snrmax) snrmax=snrt
c
450           continue
c
              snrt=(snrmax)**.5
c               
              if(type(i).eq.1) then
                do 620 j=1,np1
                   if(snrt.gt.snr1(j)) then
                     picks(pick(i))=picku1(j)
                     pnum1(j)=pnum1(j)+1
                     go to 420
                   end if
620             continue
                picks(pick(i))=picku1(np1)
                pnum1(np1)=pnum1(np1)+1
              else
                do 630 j=1,np2
                   if(snrt.gt.snr2(j)) then
                     picks(pick(i))=picku2(j)
                     pnum2(j)=pnum2(j)+1
                     go to 420
                   end if
630             continue
                picks(pick(i))=picku2(np2)
                pnum2(np2)=pnum2(np2)+1
              end if
c
420           numunc(i)=numunc(i)+1
              uncave(i)=uncave(i)+picks(pick(i))
			  snrave(i)=snrave(i)+snrt
              nsnr=nsnr+1
              nsnrt=nsnrt+1
              snrsum=snrsum+snrt
              snrsumt=snrsumt+snrt
c
            else
              picks(pick(i))=0.
            end if
c             
340      continue
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
      if(nsnr.gt.0) then
      write(6,2) ipro,it2(ipro)-it1(ipro)+1,nsnr,snrsum/nsnr
      write(11,2) ipro,it2(ipro)-it1(ipro)+1,nsnr,snrsum/nsnr
2     format(i6,2i10,f10.3)
      else
      write(6,2) ipro,it2(ipro)-it1(ipro)+1,nsnr,0.
      write(11,2) ipro,it2(ipro)-it1(ipro)+1,nsnr,0.
      end if
10    continue
c
      close(10)
c
      write(6,45)
	  write(11,45)
45	  format(/'  pick    number   ave_unc  ave_snr'/
     +                  '--------------------------'/)
c     
      do 510 i=1,npicko
         if(numunc(i).gt.0) then
		 uncave(i)=uncave(i)/float(numunc(i))
		 snrave(i)=snrave(i)/float(numunc(i))
		 end if
510   continue
c
      write(6,55)(pick(i),numunc(i),uncave(i),snrave(i),i=1,npicko)
      write(11,55)(pick(i),numunc(i),uncave(i),snrave(i),i=1,npicko)
55	  format(i6,i10,f10.4,f10.4)

c
      write(6,66)
      write(11,66)
66      format(/' pick unc  num picks'/
     +                  '--------------------')
      write(6,225) (picku1(i),pnum1(i),i=1,np1)
      write(11,225) (picku1(i),pnum1(i),i=1,np1)
      write(6,225) (picku2(i),pnum2(i),i=1,np2)
      write(11,225) (picku2(i),pnum2(i),i=1,np2)
225   format(f10.3,i10)
c
      if(nsnrt.gt.0) then
        write(6,3) nsnrt,snrsumt/nsnrt,ntraces
        write(11,3) nsnrt,snrsumt/nsnrt,ntraces
3       format(/'total number of picks: ',i10,'   ave SNR: ',f10.3/
     +          'total number of traces:',i10)
      end if
c     
      stop
      end
c
c     -------------------------------------------------------------------
c

