c
c     -----------------------------------------------------------------
c
      subroutine openf(dfile)
c
c     open data file to find its record length and re-open
c
      include 'zplot.par'
      integer fhead(13),thead(22+npickw),sint,tstart,tend
      real frhead(13)
      character dfile*80,fmt*10
      equivalence (ntraces ,fhead(1)),(npts, fhead(2)),
     +            (sint, fhead(3)),(tstart,fhead(4)),
     +            (tend,fhead(5)),(nrec,fhead(6)),
     +            (npick,fhead(7)),(vredf,frhead(8)),
     +            (ifmt,fhead(9)),(xlatlong,frhead(10)),
     +            (xelev,frhead(11)),(xutm,frhead(12)),
     +            (cm,frhead(13)),(fhead,frhead)
      common/hed/fhead,thead,nbytes
c
      open(10,file=dfile,status='old',access='direct',recl=52)
c
      read(10,rec=1) fhead
c
      if(ifmt.eq.1) then
        nbytes=npts*4+(22+npick)*4
        fmt='    real*4'
      else
        nbytes=npts*2+(22+npick)*4
        fmt=' integer*2'
      end if
c
      write(6,10) dfile,ntraces,npts,float(sint)/1000000.,
     +             float(tstart)/1000.,float(tend)/1000.,
     +             nrec,npick,vredf,fmt
10    format(/'data file opened: '/,
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
      close(10)
c
      if(vredf.lt.0..or.vredf.ge.10.) then
        write(*,*) '***  reducing velocity should be between 0 and 10
     + km/s  **'
        stop
      end if
c
      open(10,file=dfile,access='direct',recl=nbytes)
c
      return
      end
c
c     -------------------------------------------------------------------
c
      subroutine readhr(hfile,rfile,ilr,irfile,nshot,azimuth,
     +                  ishmin,ishmax)
c
      include 'zplot.par'
      integer ishoti(ntmax),itnf1(nsmax),itnf2(nsmax),
     +        itsn(ntmax),ireci(ntmax),iflagi(ntmax),igaini(ntmax),
     +        iposp(ntmax),ishstn(nsmax),itypei(ntmax),selev(ntmax),
     +        swdepth,relev(ntmax),sxutm(ntmax),syutm(ntmax),sz,rxutm,ryutm,rz
      real offsti(ntmax),picks(ntmax,npickw),pcorr(ntmax),xmod(nsmax),
     +     offstm(ntmax),offstp(ntmax),azi(ntmax),az(nsmax),azim(ntmax),
     +     frhead(13),ymod(nsmax),apicks(ntmax,npickw),
     +     rdepth(nsmax),sdepth(nsmax)
      integer fhead(13),thead(22+npickw),sint,tstart,tend
      character hfile*80,rfile*80,title*80
      equivalence (ntraces ,fhead(1)),(npts, fhead(2)),
     +            (sint, fhead(3)),(tstart,fhead(4)),
     +            (tend,fhead(5)),(nrec,fhead(6)),
     +            (npick,fhead(7)),(vredf,frhead(8)),
     +            (ifmt,fhead(9)),(xlatlong,frhead(10)),
     +            (xelev,frhead(11)),(xutm,frhead(12)),
     +            (cm,frhead(13))
      common /blk1/ ishoti,itsn,ireci,iflagi,offsti,igaini,picks,apicks,
     +              iposp,pcorr,offstm,offstp,azi,azim,itypei
      common /blk3/ itnf1,itnf2,ishstn,xmod,ymod,az,rdepth,sdepth
      common /hed/ fhead,thead,nbytes
c
      open(13, file=rfile, status='old', err=99, iostat=irfile)
c
99    if(irfile.gt.0) then
393     format(/'***  there is no record file  ***')
        write(*,*) char(7)
      end if
c
      open(16, file=hfile, form='unformatted', status='old',
     +     err=98, iostat=ihfile)
c
98    if(ihfile.gt.0) then
        write(6,393)
        write(6,394)
394     format(/'***  there is no header file  ***')
        write(*,*) char(7)
      end if
c
      write(6,338)
338   format(/
     +'record  rec#  shot stn  ntraces  trace1  trace2',
     +'  sdepth rdepth   xmod   title     '/
     +'-----------------------------------------------',
     +'----------------------')
c
      nshot=0
c
      do 20 i=1,ntraces
c
         if(ihfile.gt.0) then
           ishoti(i)=1
           itsn(i)=i
           ireci(i)=i
           itypei(i)=1
           iflagi(i)=1
           offsti(i)=1.
           azi(i)=0.
           igaini(i)=1
           do 120 j=1,npick
              picks(i,j)=0.
120        continue
         else
           read(16,end=999) ishoti(i),itsn(i),ireci(i),
     +       itypei(i),iflagi(i),offsti(i),azi(i),igaini(i),
     +       texact,slat,slong,selev(i),swdepth,rlat,rlong,relev(i),
     +       sxutm(i),syutm(i),sz,rxutm,ryutm,rz,
     +       (picks(i,j),j=1,npick)
         end if
c
         offsti(i)=offsti(i)/1000.
         azi(i)=azi(i)/60.
         azim(i)=azi(i)
         if(azim(i).gt.180.) azim(i)=azim(i)-360.
c
         if(itsn(i).eq.1) then
           if(nshot.gt.0) itnf2(nshot)=i-1
           nshot=nshot+1
           itnf1(nshot)=i
           ishstn(nshot)=ishoti(i)
		   rdepth(nshot)=relev(i)/1000.
           sdepth(nshot)=selev(i)/1000. 
         end if
c
20    continue
      
c
      itnf2(nshot)=ntraces
      ishmin=999999
      ishmax=-999999
c
      do 30 i=1,nshot
c
         if(irfile.gt.0) then
           ishnum=i
           xmod(i)=0.
           ymod(i)=0.
           az(i)=0.
           title=''
         else
	 
322        read(13,*,end=998) ishnum,xmod(i),ymod(i),az(i),title
           write(6,*) 'ishnum',ishnum,ishstn(i)
           if(ishstn(i).ne.ishnum) go to 322
         end if
c
         if(ishnum.lt.ishmin) ishmin=ishnum
         if(ishnum.gt.ishmax) ishmax=ishnum
c

         write(6,337) i,ishnum,ishstn(i),itnf2(i)-itnf1(i)+1,
     +                itnf1(i),itnf2(i),sdepth(i),rdepth(i),xmod(i),
     +                title(1:20)	 
337      format(2i6,i10,i9,2i8,1x,2f7.3,1x,f7.2,3x,a20)
30    continue
c
      rewind(13)
c
      irec=1
      do 40 i=1,ntraces
         if(ilr.gt.0.and.ireci(i).lt.ishoti(i)) offsti(i)=-offsti(i)
         if(ilr.lt.0.and.ireci(i).gt.ishoti(i)) offsti(i)=-offsti(i)
         if(ilr.eq.0) then
           if(ishoti(i).ne.ishstn(irec)) irec=irec+1
           if(azimuth.ge.-360.and.azimuth.le.360) az(irec)=azimuth
           daz=abs(az(irec)-azi(i))
           if(daz.gt.90..and.daz.lt.270.) offsti(i)=-offsti(i)
         end if
40    continue
c
      irec=1
      do 437 i=1,ntraces
         if(ishoti(i).ne.ishstn(irec)) irec=irec+1
         offstm(i)=offsti(i)+xmod(irec)
c         offstm(i)=((sxutm(i)/1000.-xmod(irec))**2+
c     +              (syutm(i)/1000.-ymod(irec))**2)**.5
c      write(*,*) offsti(i),offstm(i),xmod(irec)
437   continue
c
      return
c
998   write(*,*)'*** premature EOF encountered - rfile  ***'
      stop
c
999   write(*,*)'*** premature EOF encountered - hfile  ***'
c
      stop
      end
c
c     -------------------------------------------------------------------
c
      subroutine getsei(itn,seis,sints,iflag1,xmin,xmax,tmin,tmax,
     +           rvred,rvredf,ishotp,tminf,tmaxf,icnt,nskip,nplt,
     +           npts,gain,iplt,np1,np2,iflags,itype,icomp,ixaxis,
     +           iscreen,ifmt,ntt,ilr,ialign,talign,apick)
c
c     Determine if trace should be plotted, and if so, read that portion
c     to be plotted into the array seis from the array trace or itrace
c
      include 'zplot.par'
      integer ishoti(ntmax),ireci(ntmax),itypei(ntmax),gain,apick,
     +        iflagi(ntmax),iposp(ntmax),itsn(ntmax),igaini(ntmax),
     +        relev(ntmax),selev(ntmax)
      integer*2 itrace(npmax)
      real seis(npmax),offsti(ntmax),offstm(ntmax),
     +     picks(ntmax,npickw),pcorr(ntmax),offstp(ntmax),azi(ntmax),
     +     azim(ntmax),trace(npmax),apicks(ntmax,npickw)
      common /blk1/ ishoti,itsn,ireci,iflagi,offsti,igaini,picks,apicks,
     +              iposp,pcorr,offstm,offstp,azi,azim,itypei
c
c       if(itype.ne.0.and.itypei(itn).ne.itype) go to 10
 	  if(itype.eq.-1) then
 	    if(itypei(itn).ne.1.and.itypei(itn).ne.2) go to 10
 	  else if(itype.eq.-2) then
 	    if(itypei(itn).ne.2.and.itypei(itn).ne.3) go to 10
 	  else if(itype.eq.-3) then
	    if(itypei(itn).ne.2.and.itypei(itn).ne.4) go to 10
	  else if(itype.eq.-4) then
	    if(itypei(itn).ne.1.and.itypei(itn).ne.4) go to 10
	  else if(itype.gt.0) then
 	    if(itypei(itn).ne.itype) go to 10
	  else
	    continue
	  end if

	  
      if(icomp.ne.999.and.abs(iflagi(itn)).ne.icomp) go to 10
c
      if(abs(ixaxis).le.1) offstp(itn)=offsti(itn)
      if(abs(ixaxis).eq.2) offstp(itn)=offstm(itn)
      if(abs(ixaxis).eq.3) offstp(itn)=azi(itn)
      if(abs(ixaxis).eq.4) offstp(itn)=azim(itn)
      if(abs(ixaxis).eq.5) then
        if(ilr.ge.0) then
          offstp(itn)=float(itsn(itn))
        else
          offstp(itn)=float(ntt-itsn(itn)+1)
        end if
      end if

      gain=igaini(itn)
      if(gain.lt.1) gain=1
c
      if(iflag1.eq.0) then
        if(ishoti(itn).ne.ishotp.or.offstp(itn).lt.xmin.
     +     or.offstp(itn).gt.xmax) go to 10
c
        t1=tmin+abs(offsti(itn))*(rvred-rvredf)
        t2=tmax+abs(offsti(itn))*(rvred-rvredf)
		
        if(ialign.eq.1) then
          if(picks(itn,apick).gt.-998..and.picks(itn,apick).ne.0.)then
            t1=t1-talign+(picks(itn,apick)-
     +         abs(offsti(itn))*(rvred))
            t2=t2-talign+(picks(itn,apick)-
     +         abs(offsti(itn))*(rvred))
          else
		   if(apicks(itn,apick).gt.-998..and.apicks(itn,apick).ne.0.)then
            t1=t1-talign+(apicks(itn,apick)-
     +         abs(offsti(itn))*(rvred))
            t2=t2-talign+(apicks(itn,apick)-
     +         abs(offsti(itn))*(rvred))
		   else
           go to 10
           end if
		  end if
        end if
c
        if(t2.le.tminf.or.t1.ge.tmaxf) go to 10
c
        icnt=icnt+1
        if(mod(icnt-1,nskip).ne.0) go to 10
        iplt=iplt+1
        iposp(iplt)=itn
        if(iscreen.eq.1) write(6,65)
     +    iplt,itn,ishoti(itn),ireci(itn),offstp(itn),gain
65      format(4i10,f10.3,i12)
      end if
c
      if(iflag1.eq.0.and.iflagi(itn).le.0) go to 11
c
      if(iflag1.ne.0) then
        t1=tmin+abs(offsti(itn))*(rvred-rvredf)
        if(ialign.eq.1) then
          if(picks(itn,apick).gt.-998..and.picks(itn,apick).ne.0.) then
            t1=t1-talign+(picks(itn,apick)-
     +         abs(offsti(itn))*(rvred))
          else
            go to 10
          end if
        end if
      end if
c
      if(t1.ge.tminf) then
        nzb=0
        n1=nint((t1-tminf)/sints)+1
        pcorr(itn)=float(n1-1)*sints-t1+tminf
      else
        nzb=nint((tminf-t1)/sints)
        n1=1
        pcorr(itn)=tminf-t1-float(nzb)*sints
      end if
      n2=nplt+n1-nzb-1
      if(n2.gt.npts) then
        n2=npts
        nza=nplt-n2+n1-1-nzb
      else
        nza=0
      end if
c
      ncheck=0
      if(nzb.gt.0) then
        do 110 i=1,nzb
           ncheck=ncheck+1
           seis(ncheck)=0.
110     continue
      end if
      np1=ncheck+1
      if(ifmt.eq.1) then
c
        call gettrc(itn,trace)
c
        do 120 i=n1,n2
           ncheck=ncheck+1
           seis(ncheck)=trace(i)
120     continue
      else
c
        call gettrci(itn,itrace)
c
        do 121 i=n1,n2
           ncheck=ncheck+1
           seis(ncheck)=itrace(i)
121     continue
      end if
      np2=ncheck
      if(nza.gt.0) then
        do 130 i=1,nza
           ncheck=ncheck+1
           seis(ncheck)=0.
130     continue
      end if
c
      if(abs(ncheck-nplt).gt.0) then
        write(*,*)'***  something did not add up  ***'
        stop
      end if
c
11    iflags=1
      return
c
10    iflags=0
      return
      end
c
c     -------------------------------------------------------------------
c
      subroutine gettrc(itn,trace)
c
c     read in real*4 trace from data file
c
c     itn - trace sequential number
c
      include 'zplot.par'
      integer currec,fhead(13),thead(22+npickw)
      real trace(npmax),frhead(13)
      equivalence (npts,fhead(2)),(npick,fhead(7)),(fhead,frhead)
c
      common/hed/fhead,thead,nbytes
c
      currec=itn+1
c
      read(10,rec=currec) (thead(i),i=1,22+npick),(trace(i),i=1,npts)
c
      return
      end
c
c     -------------------------------------------------------------------
c
      subroutine gettrci(itn,itrace)
c
c     read in integer*2 trace from data file
c
c     itn - trace sequential number
c
      include 'zplot.par'
      integer currec,fhead(13),thead(22+npickw)
      integer*2 itrace(npmax)
      real frhead(13)
      equivalence (npts,fhead(2)),(npick,fhead(7)),(fhead,frhead)
c
      common/hed/fhead,thead,nbytes
c
      currec=itn+1
c
      read(10,rec=currec) (thead(i),i=1,22+npick),(itrace(i),i=1,npts)
c
      return
      end
