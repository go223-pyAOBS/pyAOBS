c
      program sgy2z
c
c     reads in a disk file in segy format and outputs z-format data for
c     plotting with zplot
c
c     Input:
c
c     The headers on the disk file are assumed to be the standard Seg_y
c     arrangement of short ands long integers. The number of samples per trace
c     is read from bytes 21,22 of the binary reel header. The number of traces
c     in the file is derived from adding the values in bytes 13,14 and 15,16
c     of the binary reel header. If you suspect these values are not
c     correct then <npts0> and <no_trace> may be re-set.
c     The number of bytes per sample is also required - if the first header
c     appears correctly but subsequent ones are wrong then change this value.
c     two bytes and four bytes are the probable values.
c
c
c     The data values are output as integers or reals dependent on the input.
c     For reals conversion from IBM reals to IEEE can be handled.
c     For other input data formats convert using ffc before running this program
c
c     Output:
c     1) A direct access file consisting of:
c        a file header - <ntrace>  - total number of traces in file
c                        <npts0>    - no of points per trace
c                        <sint>    - sampling interval (microseconds)
c                        <tstart>  - trace start time
c                        <tend>    - trace end time
c                        <nrec>    - no of shot records
c                        <npick>   - no of pick words
c     This file header is padded to be the same length as the following records
c
c     Trace header -   <nrec>    - shot station number
c                      <itsn>    - trace number
c                      <irec>    - receiver station number
c                      <itype>   - data type flag (1,2or3)
c                      <azi>     - receiver to source azimuth (minutes)
c                      <offst>   - shot to receiver offset (metres)
c                      <igword>  - multiplicative gain factor
c                      <picks>   - arrival picks
c
c     Trace values in either integer or real format - depending on input:
c        fmt ='int' trace values are read and written as 2 byte integers.
c        fmt ='sun' trace values are read and written as 4 byte IEEE reals.
c        fmt ='ibm' trace values are read as 4 byte IBM reals and written as
c             4 byte IEEE reals.
c
      parameter(npmax=5000, npickw=10, ntmax=5000, nrmax=30)
      real trace(npmax),picks(npickw),seis(npmax),seis_ibm(npmax),
     +  tstart(nrmax),tbulk(nrmax)
      integer ifile,ofile,idum4(3),tr_long1(7),tr_long2(8),resamp,
     +      tr_long3(4),sint,end_of_file,sxutm,syutm,rxutm,ryutm,omit0,
     +      itcorrect(nrmax)
      integer*2 idum2(194),tr_short1(4),tr_short2(2),tr_short3(76),
     +          itrace(npmax),iseis(npmax)
      character*80 header(40),blank,dfile,hfile
      character*1 src
      character *3 fmt,fmtw
      real*8 L0,R,ESQ,K0,lat,long,xd,yd
      COMMON /MERC/L0,R,ESQ,K0
      data tstart/nrmax*0./,itcorrect/nrmax*0/,tbulk/nrmax*0./
c
      namelist /par/ idump,tstart,nbytes,npts0,npick,src,itype,dfile,
     +               hfile,iout,nfile,vred0,vred,tmin,tmax,omit0,
     +               xmax,omax,iocalc,itcorrect,fmt,nt1,nt2,xlatlong,
     +               xelev,xutm,cm,x0,y0,rot,x00,y00,iutm,xumin,xumax,
     +               yumin,yumax,tbulk,rdepth,numread,resamp,nrecord,
     +               cfact,nchange,clat,clong
c
c     initialize
c
      nchange=0
      cfact=10000.
      nrecord=0
      omit0=0
      resamp=0
      numread=999999
      rdepth=-999999.
      xumin=99999999.
      xumax=-99999999.
      yumin=99999999.
      yumax=-99999999.
      iutm=0
      R=6.3782064D6
      ESQ=6.768657997D-3
      K0=0.9996D0
      pif=57.29577951
      cm=-999999.
      xlatlong=1.
      xelev=1.
      xutm=1.
      fmt='sun'
      iocalc=1
      omax=.01
      noffst=0
      xmax=0.
      vred0=0.
      vred=-1.
      tmin=999999.
      tmax=-999999.
      nfile=1
      iout=1
      nskip=0
      do 1 i=1,40
         header(i)=blank
1     continue
      ntrace=0
      itype=1
      ifile=20
      ofile=10
      ltrhead=88
      nrec=1
      idump=0
      nbytes=-1
      npts0=-1
      npick=5
      src='r'
      hfile='hfile'
      dfile='dfile'
      if=1
      ntfile=0
      nt1=-1
      nt2=-1
      iff=1
      nopen=0
      ishot=0
c
      open(11, file='sgy2z.in', status='old')
c
      read(11,par)
c
      if(resamp.ne.1) resamp=0
      if(iutm.eq.1.and.cm.lt.-999998.) then
        write(6,975)
975     format(/'   *** must specify central meridian  ***'/)
        stop
      else
        L0=cm
      end if
      rot=rot/pif
c
      if(iout.eq.1) open(12, file=hfile, form='unformatted')
      if(idump.gt.0) open(8, file='sgy.hdr')
      open(14, file='itype.in', status='old', err=965, iostat=itfile)
965   if(itfile.gt.0) then
        write(6,393)
393     format(/'***  there is no itype file  ***')
        itype=-abs(itype)
      end if
c
      tmaxh=tmax
      npts0h=npts0
c
1000  if(if.gt.nfile) go to 9999
c
      pos=-1.
      ishift=0
c
      call open_input (ifile)
c
      if(itfile.le.0) read(14,*) itype
c
      nopen=nopen+1
c
c     read reel header
c
      call header_input(ifile, header,3200, idum4, 12, idum2, 388)
c
      if(idump.eq.1) call dump_hd1(header,idum2,idum4,cfact)
      if(idump.eq.2) call dump_hd2(header,idum2,idum4,cfact)
c
      if(npts0h.lt.0) then
        npts0=idum2(5)
      else
        npts0=npts0h
      end if
      sint=idum2(3)
      sints=sint/1.e6
      tend=tstart(nopen)*1000.+(npts0-1)*sint*.001
      tend=tend/1000.
      if(tmin.gt.999998.) tmin=tstart(nopen)
      if(tmaxh.lt.-999998.) then
        tmax=tend
      else
        tmax=tmaxh
      end if
      itmin=nint(tmin*1000.)
      itmax=nint(tmax*1000.)
      if(vred.lt.0.) vred=vred0
      npts=nint((tmax-tmin)/sints)+1
      if(nbytes.lt.0) then
        if(idum2(7).eq.3) then
          nbytes=2
        else
          nbytes=4
        end if
      end if
      iflag=1
      do 30 j=1,npick
         picks(j)=0.0
30    continue
      nptsl=npts
      if(resamp.eq.1) nptsl=(npts-1)/2+1
      len_rec=ltrhead+(npick*4)+nptsl*nbytes
c
      if(nbytes.eq.2.and.fmt.ne.'int') then
        write(6,995) fmt
995     format(//'WARNING: 2-byte data but fmt=',a3,
     +  '  (will write as 2-byte integer)')
        fmtw='int'
      else
        fmtw=fmt
      end if
      if(nbytes.eq.4.and.fmt.eq.'int') then
        write(6,985) fmt
985     format(//'WARNING: 4-byte data but fmt=',a3)
        stop
      end if
c
      if(fmt.eq.'int') then
        ifmt=0
      else
        ifmt=1
      end if
c
      if(vred0.eq.0.) then
        rvred0=0.
      else
        rvred0=1./vred0
      end if
      if(vred.eq.0.) then
        rvred=0.
      else
        rvred=1./vred
      end if
c
      write(6,155) nopen,nbytes,npts0,sint/1000.,tstart(nopen),tend,
     +             src,npick,abs(itype),fmtw,dfile,hfile
155   format(/'file number opened:          ',i10/
     +        'number of bytes per sample:  ',i10/
     +        'number of samples per trace: ',i10/
     +        'sampling rate (millisecs):   ',f10.3/
     +        'trace start time (s):        ',f10.3/
     +        'trace end time (s):          ',f10.3/
     +        'common gather type:          ',a10/
     +        'number of pick words:        ',i10/
     +        'data type flag:              ',i10/
     +        'input data format:           ',a10/
     +        'output data file name:       ',a30/
     +        'output header file name:     ',a30/)
c
      open( ofile, file=dfile, access='direct',recl=len_rec)
c
20    continue
c
      if(nbytes.eq.2) then
         call trace_input(ifile, tr_long1, 28, tr_short1, 8,
     +                    tr_long2, 32, tr_short2, 4,
     +                    tr_long3, 16, tr_short3, 152,
     +                    itrace, nbytes*npts0  )
      else if(nbytes.eq.4) then
         call trace_input(ifile, tr_long1, 28, tr_short1, 8,
     +                    tr_long2, 32, tr_short2, 4,
     +                    tr_long3, 16, tr_short3, 152,
     +                    trace, nbytes*npts0  )
      end if
c
      if(end_of_file(ifile).ne.0) go to 9998
cc
      if(ntrace.eq.numread) go to 9998
c
      if(tr_long3(3).gt.0) tr_long3(3)=-tr_long3(3)
c
      if(iutm.eq.1) then
c
        if(nopen.eq.nchange) then
          tr_long3(2)=clat*cfact
          tr_long3(1)=clong*cfact
        end if
c
        lat=tr_long3(2)/cfact
        long=tr_long3(1)/cfact
c
c       the following xd,yd can be switched depending on the data set
c
        call geocar(lat,long,yd,xd)
c       call geocar(lat,long,xd,yd)
c
        sxutm=nint((xd-x0)*cos(rot)-(yd-y0)*sin(rot)+x00)
        syutm=nint((xd-x0)*sin(rot)+(yd-y0)*cos(rot)+y00)
c
        if(rdepth.gt.-999998.) tr_long2(8)=rdepth
c
        lat=tr_long3(4)/cfact
        long=tr_long3(3)/cfact
c
c       the following xd,yd can be switched depending on the data set
c
        call geocar(lat,long,yd,xd)
c       call geocar(lat,long,xd,yd)
c
        rxutm=nint((xd-x0)*cos(rot)-(yd-y0)*sin(rot)+x00)
        ryutm=nint((xd-x0)*sin(rot)+(yd-y0)*cos(rot)+y00)
c
        if(src.ne.'r'.and.src.ne.'R') then
          if(rxutm.lt.xumin.or.rxutm.gt.xumax.or.
     +       ryutm.lt.yumin.or.ryutm.gt.yumax) then
            nskip=nskip+1
            go to 20
          end if
        end if
        if(src.eq.'r'.or.src.eq.'R') then
          if(sxutm.lt.xumin.or.sxutm.gt.xumax.or.
     +       syutm.lt.yumin.or.syutm.gt.yumax) then
            nskip=nskip+1
            go to 20
          end if
        end if
      else
        sxutm=0
        syutm=0
        rxutm=0
        ryutm=0
      end if
c
      ntfile=ntfile+1
      if(nt1.gt.0.and.ntfile.lt.nt1) then
        nskip=nskip+1
        go to 20
      end if
      if(nt2.gt.0.and.ntfile.gt.nt2) go to 9998
c
      if(idump.eq.1) call dump_trhd1(tr_long1,tr_short1,
     +               tr_long2,tr_short2,tr_long3,tr_short3,cfact)
      if(idump.eq.2) call dump_trhd2(tr_long1,tr_short1,
     +               tr_long2,tr_short2,tr_long3,tr_short3,cfact)
c
      offst=tr_long2(1)
c
      rla=tr_long3(4)/cfact
      rlo=tr_long3(3)/cfact
      sla=tr_long3(2)/cfact
      slo=tr_long3(1)/cfact
c
      if(src.ne.'r'.and.src.ne.'R') then
        call distaz(rla,rlo,sla,slo,coffst,azi)
      else
        call distaz(sla,slo,rla,rlo,coffst,azi)
      end if
c
      if(abs(abs(offst/1000.)-coffst).gt.omax) noffst=noffst+1
      if(iocalc.eq.0) then
        coffst=offst/1000.
      else
        coffst=coffst*sign(1.,offst)
      end if
c
      if(xmax.ne.0..and.abs(coffst).gt.xmax) then
        nskip=nskip+1
        go to 20
      end if
c
c     count number of shot records and set shot station and
c     receiver station number
c
      if(src.ne.'r'.and.src.ne.'R') then
c
c       common source: shot number is source number
c
        pos1=sqrt(tr_long3(1)**2.+tr_long3(2)**2.)
c
c       if source position has changed increment nrec
c
        if(((pos1.lt.pos-1..or.pos1.gt.pos+1.).and.itype.le.0)
     +     .or.(nrecord.gt.0.and.itsn.eq.nrecord)) then
          if(itsn.gt.0) nrec=nrec+1
          itsn=0
        end if
c
      end if
c
      if(src.eq.'r'.or.src.eq.'R') then
c
c       common receiver: receiver number is source number
c
        irec=tr_long1(5)
        pos1=sqrt(tr_long3(3)**2.+tr_long3(4)**2.)
c
c       if receiver position has changed increment nrec
c
        if(((pos1.lt.pos-1..or.pos1.gt.pos+1.).and.itype.le.0)
     +     .or.(nrecord.gt.0.and.itsn.eq.nrecord)) then
          if(itsn.gt.0) nrec=nrec+1
          itsn=0
        end if
      end if
c
      pos=pos1
c
      igword=tr_short3(17)
      if(iff.eq.1) then
        write(41,555) if,tr_short3(11)
555     format(i10,i10)
        iff=0
       end if
c
      if(iout.eq.1) then
c
        t1=tmin+abs(coffst)*(rvred-rvred0)
        if(itcorrect(nopen).eq.1) t1=t1+tr_short3(11)/1000.
        if(itcorrect(nopen).eq.-1) t1=t1-tr_short3(11)/1000.
        t1=t1+tbulk(nopen)
c
        if(t1.ge.tstart(nopen)) then
          nzb=0
          n1=nint((t1-tstart(nopen))/sints)+1
        else
          nzb=nint((tstart(nopen)-t1)/sints)
          n1=1
        end if
        n2=npts+n1-nzb-1
        if(n2.gt.npts0) then
          n2=npts0
          nza=npts-n2+n1-1-nzb
        else
          nza=0
        end if
c
        if(n1.gt.npts0.or.nzb.gt.npts.or.n2.lt.1.or.nza.gt.npts.or.
     +     n1.ge.n2) then
          write(0,*) '***  something is wrong  ***'
          write(0,*) ntrace+1,nzb,n1,n2,nza,npts0,npts,coffst
          nskip=nskip+1
          go to 20
        end if
c
        ncheck=0
        if(nzb.gt.0) then
          if(nbytes.eq.4) then
            do 110 i=1,nzb
               ncheck=ncheck+1
               seis(ncheck)=0.
110         continue
          else
            do 111 i=1,nzb
               ncheck=ncheck+1
               iseis(ncheck)=0
111         continue
          end if
        end if
        np1=ncheck+1
        if(nbytes.eq.4) then
          do 120 i=n1,n2
             ncheck=ncheck+1
             seis(ncheck)=trace(i)
120       continue
        else
          do 121 i=n1,n2
             ncheck=ncheck+1
             iseis(ncheck)=itrace(i)
121       continue
        end if
        np2=ncheck
        if(nza.gt.0) then
          if(nbytes.eq.4) then
            do 130 i=1,nza
               ncheck=ncheck+1
               seis(ncheck)=0.
130         continue
          else
            do 131 i=1,nza
               ncheck=ncheck+1
               iseis(ncheck)=0
131         continue
          end if
        end if
c
        if(omit0.eq.1) then
          trmin=1.e70
          trmax=-1.e70
          if(nbytes.eq.4) then
            do 1010 j=1,npts
               if(seis(j).lt.trmin) trmin=seis(j)
               if(seis(j).gt.trmax) trmax=seis(j)
1010        continue
          else
            do 1020 j=1,npts
               if(iseis(j).lt.trmin) trmin=iseis(j)
               if(iseis(j).gt.trmax) trmax=iseis(j)
1020        continue
          end if
          if(trmin.eq.trmax) then
            nskip=nskip+1
            go to 20
          end if
        end if
c
        if(abs(ncheck-npts).gt.0) then
          write(0,*) '***  something did not add up  ***'
          write(0,*) ntrace+1,nzb,n1,n2,nza,npts0,npts,coffst
          stop
        end if
c
        coffst=coffst*1000.
        azi=azi*60.
        if(nzb.gt.0) then
          texact=tstart(nopen)-float(nzb)*sints
        else
          texact=tstart(nopen)+float(n1-1)*sints
        end if
c
        itsn=itsn+1
c
        icount=resamp+1
c
        if(nbytes.eq.2) then
          write(ofile, rec=ntrace+2) nrec,itsn,irec,
     +    abs(itype),iflag,coffst,azi,igword,texact,
     +    tr_long3(2)/cfact,tr_long3(1)/cfact,tr_long2(3),
     +    tr_long2(7)+tr_long2(3),
     +    tr_long3(4)/cfact,tr_long3(3)/cfact,tr_long2(8),
     +    sxutm,syutm,tr_long2(3),rxutm,ryutm,tr_long2(8),
     +    (picks(j),j=1,npick),(iseis(j),j=1,npts,icount)
        else if(nbytes.eq.4) then
          if(fmt.eq.'ibm') then
c
            call ibm_to_ieee_32(seis,seis_ibm,npts)
c
            write(ofile, rec=ntrace+2) nrec,itsn,irec,
     +      abs(itype),iflag,coffst,azi,igword,texact,
     +      tr_long3(2)/cfact,tr_long3(1)/cfact,tr_long2(3),
     +      tr_long2(7)+tr_long2(3),
     +      tr_long3(4)/cfact,tr_long3(3)/cfact,tr_long2(8),
     +      sxutm,syutm,tr_long2(3),rxutm,ryutm,tr_long2(8),
     +      (picks(j),j=1,npick),(seis_ibm(j),j=1,npts,icount)
          else
            write(ofile, rec=ntrace+2) nrec,itsn,irec,
     +      abs(itype),iflag,coffst,azi,igword,texact,
     +      tr_long3(2)/cfact,tr_long3(1)/cfact,tr_long2(3),
     +      tr_long2(7)+tr_long2(3),
     +      tr_long3(4)/cfact,tr_long3(3)/cfact,tr_long2(8),
     +      sxutm,syutm,tr_long2(3),rxutm,ryutm,tr_long2(8),
     +      (picks(j),j=1,npick),(seis(j),j=1,npts,icount)
          end if
        end if
c
        write(12)  nrec,itsn,irec,abs(itype),iflag,coffst,
     +     azi,igword,texact,
     +     tr_long3(2)/cfact,tr_long3(1)/cfact,tr_long2(3),
     +     tr_long2(7)+tr_long2(3),
     +     tr_long3(4)/cfact,tr_long3(3)/cfact,tr_long2(8),
     +     sxutm,syutm,tr_long2(3),rxutm,ryutm,tr_long2(8),
     +     (picks(j),j=1,npick)
      end if
c
      ntrace=ntrace+1
      go to 20
c
9998  continue
c
      call close_input(ifile)
c
      if=if+1
      iff=1
      go to 1000
c
c     write header
c
9999  if(iout.eq.1) then
        if(resamp.eq.1) then
          npts=(npts-1)/2+1
          sint=sint*2.
        end if
c
        write(ofile,rec=1) ntrace,npts,sint,itmin,itmax,nrec,npick,
     +                     vred,ifmt,xlatlong,xelev,xutm,cm
      end if
c
      write(6,165) ntfile,ntrace,nrec,nskip,noffst
165   format('number of traces read:                 ',i10/
     +       'number of traces written:              ',i10/
     +       'number of records written:             ',i10/
     +       'number of traces skipped:              ',i10/
     +       'number of traces with incorrect offset:',i10/)
c
      close(ofile)
c
      stop
      end
